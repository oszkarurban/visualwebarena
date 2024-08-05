import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig
import torch.nn as nn

from torchvision import transforms

from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union, Iterable

from fastchat.model import get_conversation_template

import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from torchvision.transforms.functional import to_pil_image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from typing import Dict, List, Optional, Union
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)

import PIL
from transformers.utils.generic import TensorType
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import ChannelDimension

"""
internvl_chat/internvl/conversation.py

Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have any changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

"""
/home/ubuntu/.cache/huggingface/modules/transformers_modules/OpenGVLab/Mini-InternVL-Chat-4B-V1-5/5abc8a829e1c848bcb7cc79f22a70e073f68ba87/modeling_internvl_chat.py
/home/ubuntu/miniforge3/envs/internvl/lib/python3.9/site-packages/transformers/generation/utils.py

commented(removed)  #@torch.no_grad()
"""


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def build_transform_pgd(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
 #       T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
 #       T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_preprocess_pgd(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = list(image.size()[1:])
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = T.Resize((target_width, target_height))(image)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        #resized_img.crop(box)
        """
        box – The crop rectangle, as a (left, upper, right, lower)-tuple.
        The right can also be represented as (left+width)
        and lower can be represented as (upper+height).
        """

        #(img: Tensor, top: int, left: int, height: int, width: int)
        #height = abs(upper-lower) = abs(box[1]-box[3])
        #width = abs(right-left) = abs(box[2]-box[0])
        split_img = T.functional.crop(resized_img, box[1],box[0],np.abs(box[1]-box[3]),np.abs(box[2]-box[0])) 
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = T.Resize((image_size, image_size))(image)
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_image_from_tensor(image_tensor, input_size=448, max_num=6): #the input is an image tensor not a PIL, comnpared to load_image
    #to_pil = transforms.ToPILImage()
    #image = to_pil(image_tensor.squeeze(0))

    transform = build_transform_pgd(input_size)
    images = dynamic_preprocess_pgd(image_tensor, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    pixel_values = transform(image_tensor)
    return pixel_values

def tensor_load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    new_size = (input_size, input_size) 

    # resize the image
    #resized_image = image.resize(new_size) #needs to be done explicitly because resizing is done in the orgiginal load_image
    
    #convert the image to torch.tensor
    image_tensor = transforms.ToTensor()(image)
    return image_tensor

def to_channel_dimension_format(
    image: torch.Tensor,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> torch.Tensor:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`torch.Tensor`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `torch.Tensor`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input image must be of type torch.Tensor, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.permute((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.permute((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image

def infer_channel_dimension_format(
    image: torch.Tensor, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`torch.Tensor`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST

    raise ValueError("Unable to infer channel dimension format")

def normalize(
    image: torch.Tensor,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> torch.Tensor:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`torch.Tensor`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError("image must be a torch.Tensor")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not torch.is_floating_point(image):
        image = image.to(torch.float32)

    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device)

    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = torch.tensor(std, dtype=image.dtype, device=image.device)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.permute(1, 2, 0) - mean) / std).permute(2, 0, 1)

    if data_format is not None:
        image = to_channel_dimension_format(image, data_format, input_data_format)

    return image

def get_channel_dimension_axis(image: torch.Tensor, input_data_format: Union[str, ChannelDimension]) -> int:
    if input_data_format == ChannelDimension.FIRST:
        return 0 if image.dim() == 3 else 1
    elif input_data_format == ChannelDimension.LAST:
        return -1
    else:
        raise ValueError(f"Unsupported input data format: {input_data_format}")


def preprocess(
    self,
    images: ImageInput,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: PILImageResampling = None,
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    do_convert_rgb: bool = None,
    data_format: ChannelDimension = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    **kwargs,
) -> PIL.Image.Image:
    """
    Preprocess an image or batch of images.

    Args:
        images (`ImageInput`):
            Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
            passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Controls the size of the image after `resize`. The shortest edge of the image is resized to
            `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
            is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
            edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image values between [0 - 1].
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to normalize the image by if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*):
            The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    """
    do_resize = do_resize if do_resize is not None else self.do_resize
    resample = resample if resample is not None else self.resample
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
    do_normalize = do_normalize if do_normalize is not None else self.do_normalize
    image_mean = image_mean if image_mean is not None else self.image_mean
    image_std = image_std if image_std is not None else self.image_std
    do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

    size = size if size is not None else self.size
    size = get_size_dict(size, default_to_square=False)

    images = make_list_of_images(images)

    #validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

    if not valid_images(images):
        raise ValueError(
            "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
            "torch.Tensor, tf.Tensor or jax.ndarray."
        )

    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )
    # PIL RGBA images are converted to RGB
    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]

    # All transformations expect numpy arrays.
    # images = [to_numpy_array(image) for image in images]

    #if is_scaled_image(images[0]) and do_rescale:
        # logger.warning_once(
        #     "It looks like you are trying to rescale already rescaled images. If the input"
        #     " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
        # )

    if input_data_format is None:
        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(images[0])

    transforms=[]
    if do_resize:
        transforms.append(T.Resize((size['height'], size['width']), interpolation=InterpolationMode.BICUBIC))
    if do_rescale:
        transforms.append(T.Lambda(lambda img: img * rescale_factor))
    if do_normalize:
        transforms.append(T.Normalize(mean=image_mean, std=image_std))

    transform = T.Compose(transforms)
    images = [transform(image) for image in images]
    
    images = [
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
    ]

    encoded_outputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

    return encoded_outputs

def save_adv_image(adv, path):
    if adv.is_cuda:
        adv = adv.cpu().detach()
    else:
        adv = adv.detach()
    
    # Remove the batch dimension and convert to numpy array
    save_image(adv,path)

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy

def preprocess_txt(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_encoding = None

        

        return text_encoding

def fgsm_(captioning_model, captioning_processor, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    sucess=False

    """Internal process for all FGSM and PGD attacks."""    
    input_ = x.clone().detach_().to("cuda")
    input_.requires_grad_()

    inputs = preprocess(self=captioning_processor.image_processor, return_tensors="pt",images=input_, do_rescale=False)

    text="describe the image"
    inputs_txt= preprocess_txt(self=captioning_processor,text=text,return_tensors="pt")
    if (inputs is not None) and (inputs_txt is not None):
            inputs.update(inputs_txt)
    inputs.to(torch.device("cuda"), torch.float16)

    labels = captioning_processor.tokenizer("write hello into a textfield", return_tensors="pt").input_ids
    result = captioning_model.forward(pixel_values=inputs.pixel_values,input_ids=inputs.input_ids, labels=labels.cuda())

    #res = captioning_model(pixel_values=inputs.pixel_values,input_ids=inputs.input_ids, labels=labels.cuda()) alternatively call model()
    print(f"resonse FORWARD: {captioning_processor.batch_decode(torch.argmax(result.logits, dim=2).tolist(), skip_special_tokens=True)}")


    print("GENERATE:")
    inputs_i = preprocess(self=captioning_processor.image_processor, return_tensors="pt",images=input_, do_rescale=False).to(torch.device("cuda"), torch.float16)
    generation_config = dict(
    output_logits=True,
    return_dict_in_generate=True,
    max_new_tokens=target.size()[0]
    )
    result_g = captioning_model.generate(
        **inputs_i, **generation_config
    )
    response = captioning_processor.batch_decode(
        result_g.sequences, skip_special_tokens=True
    )
    print(f"resonse GENERATE: {response}")

    loss = result.loss
    captioning_model.zero_grad()
    loss.backward()
    
    if sucess==False:
        #perfrom either targeted or untargeted attack
        if targeted:
            out = input_ - eps * input_.grad#.sign()
        else:
            out = input_ + eps * input_.grad#.sign()
    else:
        out = input_
    
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd(captioning_model, captioning_processor, x, target, k, eps, eps_step, targeted, clip_min, clip_max):
    print("NEW PGD")
    x_min = (x - eps).cuda()
    x_max = (x + eps).cuda()
    
    # Randomize the starting point x.
    x_adv = x + eps * (2 * torch.rand_like(x) - 1)
    # Clamp back
    if (clip_min is not None) or (clip_max is not None):
        x_adv.clamp_(min=clip_min, max=clip_max)
    
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None) 
        # as we want to apply the attack as defined
        x_adv = fgsm_(captioning_model, captioning_processor, x_adv, target,eps_step, targeted)
        # Projection Step
        x_adv = torch.min(x_max, torch.max(x_min, x_adv))
        
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x_adv.clamp_(min=clip_min, max=clip_max)
    return x_adv

