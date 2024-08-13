import os
import re
from typing import List, Tuple, Optional, Dict
import logging
from IPython.display import HTML, display
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures

# This Default Prompt Using Chinese and could be changed to other languages.

DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""
DEFAULT_RECT_PROMPT = """图片中用红色框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""
DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""


def _is_near(rect1: BaseGeometry, rect2: BaseGeometry, distance: float = 5) -> bool:
    """
    Check if two rectangles are near each other if the distance between them is less than the target.
    """
    return rect1.buffer(0.1).distance(rect2.buffer(0.1)) < distance


def _is_horizontal_near(rect1: BaseGeometry, rect2: BaseGeometry, distance: float = 5) -> bool:
    """
    Check if two rectangles are near horizontally if one of them is a horizontal line.
    """
    result = False
    if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
        if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
            result = abs(rect1.bounds[3] - rect2.bounds[3]) < distance
    return result


def _union_rects(rect1: BaseGeometry, rect2: BaseGeometry) -> BaseGeometry:
    """
    Union two rectangles.
    """
    return sg.box(*(rect1.union(rect2).bounds))


def _merge_rects(rect_list: List[BaseGeometry], distance: float = 5, horizontal_distance: Optional[float] = None,near_distance : float = 5,
                horizontal_near_distance : float = 5) -> \
        List[BaseGeometry]:
    """
    Merge rectangles in the list if the distance between them is less than the target.
    """
    merged = True
    while merged:
        merged = False
        new_rect_list = []
        while rect_list:
            rect = rect_list.pop(0)
            for other_rect in rect_list:
                if _is_near(rect, other_rect, near_distance) or (
                        horizontal_distance and _is_horizontal_near(rect, other_rect, horizontal_near_distance)):
                    rect = _union_rects(rect, other_rect)
                    rect_list.remove(other_rect)
                    merged = True
            new_rect_list.append(rect)
        rect_list = new_rect_list
    return rect_list


def _adsorb_rects_to_rects(source_rects: List[BaseGeometry], target_rects: List[BaseGeometry], distance: float = 5) -> \
        Tuple[List[BaseGeometry], List[BaseGeometry]]:
    """
    Adsorb a set of rectangles to another set of rectangles.
    """
    new_source_rects = []
    for text_area_rect in source_rects:
        adsorbed = False
        for index, rect in enumerate(target_rects):
            if _is_near(text_area_rect, rect, distance):
                rect = _union_rects(text_area_rect, rect)
                target_rects[index] = rect
                adsorbed = True
                break
        if not adsorbed:
            new_source_rects.append(text_area_rect)
    return new_source_rects, target_rects

def _parse_rects(page,
                near_distance : float = 5,
                horizontal_near_distance : float = 5,
                merge_distance : float = 20,
                horizontal_merge_distance : Optional[float] = None,
                minimun_merge_size : int = 20,) -> List[Tuple[float, float, float, float]]: #: fitz.Page
    """
    Parse drawings and images in the page and merge adjacent rectangles.
    """
    # 提取画的内容
    drawings = page.get_drawings()

    # 忽略掉长度小于30的水平直线
    is_short_line = lambda x: abs(x['rect'][3] - x['rect'][1]) < 10 and abs(x['rect'][2] - x['rect'][0]) < 10
    drawings = [drawing for drawing in drawings if not is_short_line(drawing)]

    # 转换为shapely的矩形
    rect_list = [sg.box(*drawing['rect']) for drawing in drawings]

    # 提取图片区域
    images = page.get_image_info()
    image_rects = [sg.box(*image['bbox']) for image in images]

    # 合并drawings和images
    rect_list += image_rects

    merged_rects = _merge_rects(rect_list, distance=merge_distance, horizontal_distance=horizontal_merge_distance,near_distance=near_distance,horizontal_near_distance=horizontal_near_distance)
    merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

    # 过滤比较小的矩形
    merged_rects = [rect for rect in merged_rects if rect.bounds[2] - rect.bounds[0] > minimun_merge_size and rect.bounds[3] - rect.bounds[1] > minimun_merge_size]

    return [rect.bounds for rect in merged_rects]


def _parse_pdf_to_images(pdf_path: str,
                         output_dir: str = './',
                         near_distance : float = 5,
                         horizontal_near_distance : float = 5,
                         merge_distance : float = 20,
                         horizontal_merge_distance : Optional[float] = None,
                         minimun_merge_size : int = 20,
                         finance : bool = False,) -> List[Tuple[str, List[str]]]:
    """
    Parse PDF to images and save to output_dir.
    """
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    image_infos = []
    recs_info = []
    for page_index, page in enumerate(pdf_document):
        logging.info(f'parse page: {page_index}')
        rect_images = []
        rects = _parse_rects(page,
                             near_distance=near_distance,
                             horizontal_near_distance=horizontal_near_distance,
                             merge_distance=merge_distance,
                             horizontal_merge_distance=horizontal_merge_distance,
                             minimun_merge_size=minimun_merge_size)
        recs_info.append(rects)
        for index, rect in enumerate(rects):
            if finance == True: #finance mode
                x0,y0,x1,y1 = rect
                rect = [x0-40,y0-60,x1+30,y1+40]
                fitz_rect = fitz.Rect(rect)
            else :
                fitz_rect = fitz.Rect(rect)

            # 保存页面为图片
            pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
            name = f'{page_index}_{index}.png'
            pix.save(os.path.join(output_dir, name))
            rect_images.append(name)
        page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        page_image = os.path.join(output_dir, f'{page_index}.png')
        page_image_with_rects.save(page_image)
        image_infos.append((page_image, rect_images))

    pdf_document.close()
    return image_infos,recs_info


def parse_pdf(
        pdf_path: str,
        output_dir: str = './pic',
        verbose: bool = False,
        near_distance : float = 5,
        horizontal_near_distance : float = 5,
        merge_distance : float = 20,
        horizontal_merge_distance : Optional[float] = None,
        minimun_merge_size : int = 20,
        finance : bool = False,
) -> Tuple[str, List[str]]:
    """
    Parse a PDF file to a markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos,recs_info = _parse_pdf_to_images(pdf_path, output_dir=output_dir,near_distance = near_distance,horizontal_near_distance = horizontal_near_distance,merge_distance = merge_distance,horizontal_merge_distance=horizontal_merge_distance,minimun_merge_size=minimun_merge_size,finance=finance)

    all_rect_images = []
    # remove all rect images
    if not verbose:
        for page_image, rect_images in image_infos:
            if os.path.exists(page_image):
                os.remove(page_image)
            all_rect_images.extend(rect_images)
    return all_rect_images,recs_info
    # return content, all_rect_images

def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

