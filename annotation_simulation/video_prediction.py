# code requires the SAM2 repository by Meta AI (https://github.com/facebookresearch/sam2), licensed under the Apache License 2.0

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import monai
import gc
import SimpleITK as sitk
import copy
import math
import argparse
from monai.metrics import compute_surface_dice


if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


from sam2.build_sam import build_sam2_video_predictor


def show_mask_2D(mask_2D, ax, color='red'):
    mask_2D_overlay = np.zeros((*mask_2D.shape, 4))
    mask_2D_overlay[mask_2D == 1] = np.array([1, 0, 0, 0.15]) if color == 'red' else np.array([0, 1, 0, 0.3])
    ax.imshow(mask_2D_overlay)



def calc_metrics(mask_2D, predicted_mask_2D, spacing_y, spacing_x):

    tp = np.sum((mask_2D == 1) & (predicted_mask_2D == 1))
    fp = np.sum((mask_2D == 0) & (predicted_mask_2D == 1))
    tn = np.sum((mask_2D == 0) & (predicted_mask_2D == 0))
    fn = np.sum((mask_2D == 1) & (predicted_mask_2D == 0))

    # Calculate IoU
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union != 0 else np.nan

    dice = 2 * tp / (2 * tp + fp + fn)

    # Calculate Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else np.nan

    # Calculate Recall
    recall = tp / (tp + fn) if (tp + fn) != 0 else np.nan

    label_mask_2D_tensor = torch.tensor(mask_2D).unsqueeze(0).unsqueeze(0)
    pred_mask_2D_tensor = torch.tensor(predicted_mask_2D).unsqueeze(0).unsqueeze(0)

    # hd_95 = float(monai.metrics.compute_hausdorff_distance(np.expand_dims(predicted_mask_2D, axis=(0,1)), np.expand_dims(mask_2D, axis=(0,1)), percentile=95, spacing=1))
    hd_95 = float(monai.metrics.compute_hausdorff_distance(pred_mask_2D_tensor, label_mask_2D_tensor, percentile=95, spacing=(spacing_y, spacing_x)))

    tolerance_1mm = [1.0]  # Tolerance of 1 mm
    tolerance_3mm = [3.0]  # Tolerance of 3 mm

    # Compute NSD for 1 mm tolerance globally across the whole 3D scan
    nsd_1mm = compute_surface_dice(
        y_pred=pred_mask_2D_tensor,
        y=label_mask_2D_tensor,
        class_thresholds=tolerance_1mm,
        spacing=(spacing_y, spacing_x),
        include_background=False
    ).item()

    # Compute NSD for 3 mm tolerance globally across the whole 3D scan
    nsd_3mm = compute_surface_dice(
        y_pred=pred_mask_2D_tensor,
        y=label_mask_2D_tensor,
        class_thresholds=tolerance_3mm,
        spacing=(spacing_y, spacing_x),
        include_background=False
    ).item()


    if math.isnan(hd_95) and tp == 0:
        hd_95 = np.inf

    if math.isnan(hd_95):
        raise ValueError("Hausdorff distance (hd_95) is NaN.")
    if math.isnan(iou):
        raise ValueError("IoU is NaN.")
    if np.sum((mask_2D == 1)) == 0:
        raise ValueError("Mask has no positive pixels.")


    return tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm




def create_comparison_plot(image_2D, mask_2D_predicted, mask_2D, iou, dice, precision, recall, hd_95, vis_path, out_frame_idx, label_id):
    # Create a figure with 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Original image_2D
    axes[0].imshow(image_2D, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Image with overlay predicted mask_2D
    axes[1].imshow(image_2D, cmap='gray')
    show_mask_2D(mask_2D_predicted, axes[1], color='red')
    axes[1].set_title(f"Image with Predicted Mask for label id {label_id}")
    axes[1].axis('off')

    # Image with overlay true mask_2D
    axes[2].imshow(image_2D, cmap='gray')
    show_mask_2D(mask_2D, axes[2], color='green')
    axes[2].set_title(f"Image with True Mask for label id {label_id}")
    axes[2].axis('off')

    # Image with overlay both mask_2Ds
    axes[3].imshow(image_2D, cmap='gray')
    show_mask_2D(mask_2D_predicted, axes[3], color='red')
    show_mask_2D(mask_2D, axes[3], color='green')
    axes[3].set_title(f"Image with Predicted and True Mask for label id {label_id}")
    axes[3].axis('off')

    #plt.tight_layout()
    # Add text below the panels with IoU, precision, and recall
    plt.figtext(0.5, 0.01, f"DCE: {dice:.4f}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, 'HD 95': {hd_95:.4f}", ha="center", fontsize=12)

    plt.savefig(os.path.join(vis_path, f"mask_2D_comparison_label_id_{label_id}_frame_{out_frame_idx}.png"))
    plt.close(fig)
    plt.close('all')

def process(model_type, dataset_path, patient, dice_thresh, hd_95_thresh, prompt):
    
    print(f"processing {patient} dice_thresh {dice_thresh} hd_95_thresh {hd_95_thresh} prompt {prompt}")
    script_dir = Path(__file__).resolve().parent
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if model_type == 'sam2_hiera_tiny':
            sam2_checkpoint = script_dir.parent / "checkpoints" / "sam2_hiera_tiny.pt" # see download_ckpts.sh in checkpoits folder
            model_type_desc = 'tiny'
        model_cfg = "sam2_hiera_t.yaml"
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        

        images_3D_path = dataset_path / f"SAM_2_frames_4/imagesTr/{patient}"
        masks_3D_path = dataset_path / "SAM_2_frames_4/labelsTr_cls_1/"

        labeled_masks_3D_path = dataset_path / "SAM_2_frames_4/labeled_labelsTr/"
        vis_path = dataset_path / f"SAM_2_frames_4/{model_type_desc}_vis/{patient}_dice_thresh_{'_'.join(str(dice_thresh).split('.'))}_hd_95_thresh_{'_'.join(str(hd_95_thresh).split('.'))}_prompt_{prompt}"
        masks_3D_pred_path = dataset_path / f"SAM_2_frames_4/{model_type_desc}_predictionsTr/dice_thresh_{'_'.join(str(dice_thresh).split('.'))}_hd_95_thresh_{'_'.join(str(hd_95_thresh).split('.'))}_prompt_{prompt}"
        entity_masks_3D_pred_path = dataset_path / f"SAM_2_frames_4/{model_type_desc}_entity_predictionsTr/dice_thresh_{'_'.join(str(dice_thresh).split('.'))}_hd_95_thresh_{'_'.join(str(hd_95_thresh).split('.'))}_prompt_{prompt}"

        mask_3D_annotation_simulation_path = dataset_path / f"SAM_2_frames_4/{model_type_desc}_ann_sim_labelsTr/dice_thresh_{'_'.join(str(dice_thresh).split('.'))}_hd_95_thresh_{'_'.join(str(hd_95_thresh).split('.'))}_prompt_{prompt}"
        
        
        coordinates_center = pd.read_csv(images_3D_path / 'coordinates_center.csv')

        mask_3D_sitk = sitk.ReadImage(Path(masks_3D_path) / (patient + ".nii.gz"))
        mask_3D = sitk.GetArrayFromImage(mask_3D_sitk)

        spacing_x, spacing_y, spacing_z = mask_3D_sitk.GetSpacing()


        if 1 not in mask_3D: # if class 1 is not in mask, do not segment
            torch.cuda.empty_cache()
            return {'patient': patient, 'warning': 'no class 1 in label mask volume'}
        labeled_mask_3D_sitk = sitk.ReadImage(Path(labeled_masks_3D_path) / (f'{patient}_cls_1' + ".nii.gz"))
        labeled_mask_3D = sitk.GetArrayFromImage(labeled_mask_3D_sitk)

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(images_3D_path)
            if os.path.splitext(p)[-1] in [".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        Path(vis_path).mkdir(parents=True, exist_ok=True)
        Path(masks_3D_pred_path).mkdir(parents=True, exist_ok=True)
        Path(entity_masks_3D_pred_path).mkdir(parents=True, exist_ok=True)
        Path(mask_3D_annotation_simulation_path).mkdir(parents=True, exist_ok=True)

        entity_mask_3D_pred = np.zeros_like(mask_3D)

        inference_state = predictor.init_state(video_path=images_3D_path)

        stats = []
        entity_masks_3D_annotation_simulation = []
        entity_masks_3D_pred = []
        ##############
        ## initial prompt in the center slice
        centers = coordinates_center[coordinates_center['class'] == 1]
        for index_center, center in centers.iterrows():

            predictor.reset_state(inference_state)
            
            center_z = center['center_z']
            
            # mask of one entity, e.g. one tumor
            entity_mask_3D = np.where(labeled_mask_3D == center['label_id'], 1, 0)
            entity_mask_3D_annotation_simulation = copy.deepcopy(entity_mask_3D)

            entity_mask_2D = np.where(labeled_mask_3D[center_z,:,:] == center['label_id'], 1, 0).astype('uint8')

            z_indices, y_indices, x_indices = np.where(labeled_mask_3D == center['label_id'])
            z_min_obj, z_max_obj = np.min(z_indices), np.max(z_indices)
            num_z_steps_forward = z_max_obj - center_z # only object boundaries, afterwards no sam prediction
            num_z_steps_backwards = center_z - z_min_obj
            corrected = False
            if prompt == 'mask': # use precise segmentation mask to prompt SAM 2
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=center_z,
                    obj_id=center['label_id'],
                    mask=entity_mask_2D,
                )
                entity_mask_2D_pred = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype('uint8')
                entity_mask_3D_pred[center_z] = entity_mask_2D_pred
                tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm  = calc_metrics(entity_mask_2D, entity_mask_2D_pred, spacing_y, spacing_x)

            elif prompt == 'box': # use box to prompt SAM 2
                box = np.array([center['bbox_x1'], center['bbox_y1'], center['bbox_x2'], center['bbox_y2']], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=center_z,
                    obj_id=center['label_id'],
                    box=box,
                )
                entity_mask_2D_pred = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype('uint8')
                entity_mask_3D_pred[center_z] = entity_mask_2D_pred
                tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm  = calc_metrics(entity_mask_2D, entity_mask_2D_pred, spacing_y, spacing_x)

                if (dice < dice_thresh or hd_95 > hd_95_thresh): # TODO also Haussdorf distance or NSD
                    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=center_z,
                        obj_id=center['label_id'],
                        mask=entity_mask_2D,
                    )
                    corrected = True
                else:
                    entity_mask_3D_annotation_simulation[center_z,:,:] = entity_mask_2D_pred
            image_2D = np.array(Image.open(os.path.join(images_3D_path, frame_names[center_z])))
            create_comparison_plot(image_2D, entity_mask_2D_pred, entity_mask_2D, iou, dice, precision, recall, hd_95, vis_path, center_z, center['label_id'])
            stats.append({'frame': frame_names[center_z],
                        'label id': center['label_id'],
                        'seed': True,
                        'corrected': corrected,
                        'total positives': np.sum((entity_mask_2D == 1)),
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'fn': fn,
                        'iou': iou,
                        'dice': dice,
                        'precision': precision,
                        'recall': recall,
                        'hd_95': hd_95,
                        'nsd_1mm': nsd_1mm,
                        'nsd_3mm': nsd_3mm,
                        })

            # forward prediction
            video_segments = {} 
            for z_steps_forward in range(0, num_z_steps_forward):
                # propagate video
                video_segments = {} 
                for count, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(predictor.propagate_in_video(inference_state, start_frame_idx=center_z+z_steps_forward+1, max_frame_num_to_track=0)): #, start_frame_idx=coordinates_center['center_z'].min())
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                for out_frame_idx in video_segments.keys():
                                        
                    corrected = False
                    image_2D = np.array(Image.open(os.path.join(images_3D_path, frame_names[out_frame_idx])))
                    entity_mask_2D_pred = video_segments[out_frame_idx][center['label_id']][0].astype('uint8')
                    entity_mask_3D_pred[out_frame_idx] = entity_mask_2D_pred
                    

                    entity_mask_2D = np.where(labeled_mask_3D[out_frame_idx,:,:] == center['label_id'], 1, 0)
                    tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm  = calc_metrics(entity_mask_2D, entity_mask_2D_pred, spacing_y, spacing_x)

                    create_comparison_plot(image_2D, entity_mask_2D_pred, entity_mask_2D, iou, dice, precision, recall, hd_95, vis_path, out_frame_idx, center['label_id'])

                    if (dice < dice_thresh or hd_95 > hd_95_thresh) and int(out_frame_idx) != int(center_z): # TODO also Haussdorf distance or NSD
                        predictor.reset_state(inference_state)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=center['label_id'],
                            mask=entity_mask_2D,
                        )
                        corrected = True
                    else:
                        if int(out_frame_idx) != int(center_z):
                            entity_mask_3D_annotation_simulation[out_frame_idx,:,:] = entity_mask_2D_pred

                    stats.append({'frame': frame_names[out_frame_idx],
                                'label id': center['label_id'],
                                'seed': True if int(out_frame_idx) == int(center['center_z']) else False,
                                'corrected': corrected,
                                'total positives': np.sum((entity_mask_2D == 1)),
                                'tp': tp,
                                'fp': fp,
                                'tn': tn,
                                'fn': fn,
                                'iou': iou,
                                'dice': dice,
                                'precision': precision,
                                'recall': recall,
                                'hd_95': hd_95,
                                'nsd_1mm': nsd_1mm,
                                'nsd_3mm': nsd_3mm,
                                })

            # backwards prediction
            predictor.reset_state(inference_state)
            entity_mask_2D = np.where(labeled_mask_3D[center_z,:,:] == center['label_id'], 1, 0)
            corrected = False
            if prompt == 'mask': # use precise segmentation mask to prompt SAM 2
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=center_z,
                    obj_id=center['label_id'],
                    mask=entity_mask_2D,
                )
                entity_mask_2D_pred = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype('uint8')
                entity_mask_3D_pred[center_z] = entity_mask_2D_pred
                tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm  = calc_metrics(entity_mask_2D, entity_mask_2D_pred, spacing_y, spacing_x)

            elif prompt == 'box': # use box to prompt SAM 2
                box = np.array([center['bbox_x1'], center['bbox_y1'], center['bbox_x2'], center['bbox_y2']], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=center_z,
                    obj_id=center['label_id'],
                    box=box,
                )
                entity_mask_2D_pred = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype('uint8')
                entity_mask_3D_pred[center_z] = entity_mask_2D_pred
                tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm  = calc_metrics(entity_mask_2D, entity_mask_2D_pred, spacing_y, spacing_x)

                if (dice < dice_thresh or hd_95 > hd_95_thresh): # TODO also Haussdorf distance or NSD
                    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=center_z,
                        obj_id=center['label_id'],
                        mask=entity_mask_2D,
                    )
                    corrected = True
                else:
                    entity_mask_3D_annotation_simulation[center_z,:,:] = entity_mask_2D_pred
            image_2D = np.array(Image.open(os.path.join(images_3D_path, frame_names[center_z])))
            create_comparison_plot(image_2D, entity_mask_2D_pred, entity_mask_2D, iou, dice, precision, recall, hd_95, vis_path, center_z, center['label_id'])
            stats.append({'frame': frame_names[center_z],
                        'label id': center['label_id'],
                        'seed': True,
                        'corrected': corrected,
                        'total positives': np.sum((entity_mask_2D == 1)),
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'fn': fn,
                        'iou': iou,
                        'dice': dice,
                        'precision': precision,
                        'recall': recall,
                        'hd_95': hd_95,
                        'nsd_1mm': nsd_1mm,
                        'nsd_3mm': nsd_3mm,
                        })
            
            for z_steps_backwards in range(0, num_z_steps_backwards):
                video_segments = {} 
                # propagate video
                for count, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(predictor.propagate_in_video(inference_state, start_frame_idx=center_z-z_steps_backwards-1, max_frame_num_to_track=0, reverse=True)): #, start_frame_idx=coordinates_center['center_z'].min())
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                for out_frame_idx in video_segments.keys():
                                            
                    corrected = False
                    image_2D = np.array(Image.open(os.path.join(images_3D_path, frame_names[out_frame_idx])))
                    entity_mask_2D_pred = video_segments[out_frame_idx][center['label_id']][0].astype('uint8')
                    entity_mask_3D_pred[out_frame_idx] = entity_mask_2D_pred


                    entity_mask_2D = np.where(labeled_mask_3D[out_frame_idx,:,:] == center['label_id'], 1, 0)
                    tp, fp, tn, fn, iou, dice, precision, recall, hd_95, nsd_1mm, nsd_3mm = calc_metrics(entity_mask_2D, entity_mask_2D_pred, spacing_y, spacing_x)
                    
                    create_comparison_plot(image_2D, entity_mask_2D_pred, entity_mask_2D, iou, dice, precision, recall, hd_95, vis_path, out_frame_idx, center['label_id'])

                    if (dice < dice_thresh or hd_95 > hd_95_thresh) and int(out_frame_idx) != int(center_z): # TODO also Haussdorf distance or NSD
                        predictor.reset_state(inference_state)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=center['label_id'],
                            mask=entity_mask_2D,
                        )
                        corrected = True
                    else:
                        if int(out_frame_idx) != int(center_z):
                            entity_mask_3D_annotation_simulation[out_frame_idx,:,:] = entity_mask_2D_pred

                    stats.append({'frame': frame_names[out_frame_idx],
                                'label id': center['label_id'],
                                'seed': True if int(out_frame_idx) == int(center['center_z']) else False,
                                'corrected': corrected,
                                'total positives': np.sum((entity_mask_2D == 1)),
                                'tp': tp,
                                'fp': fp,
                                'tn': tn,
                                'fn': fn,
                                'iou': iou,
                                'dice': dice,
                                'precision': precision,
                                'recall': recall,
                                'hd_95': hd_95,
                                'nsd_1mm': nsd_1mm,
                                'nsd_3mm': nsd_3mm,
                                })
            entity_masks_3D_annotation_simulation.append(entity_mask_3D_annotation_simulation)
            entity_masks_3D_pred.append(entity_mask_3D_pred)

            entity_mask_3D_pred_sitk = sitk.GetImageFromArray(entity_mask_3D_pred)
            entity_mask_3D_pred_sitk.SetOrigin(mask_3D_sitk.GetOrigin())
            entity_mask_3D_pred_sitk.SetSpacing(mask_3D_sitk.GetSpacing())
            entity_mask_3D_pred_sitk.SetDirection(mask_3D_sitk.GetDirection())
            output_path = Path(entity_masks_3D_pred_path) / f"{patient}_label_id_{center['label_id']}.nii.gz"
            sitk.WriteImage(entity_mask_3D_pred_sitk, output_path)

        pd.DataFrame(stats).to_csv(Path(vis_path) / 'stats.csv')

        mask_3D_annotation_simulation = np.zeros_like(entity_mask_3D)
        for e in entity_masks_3D_annotation_simulation:
            mask_3D_annotation_simulation = np.logical_or(mask_3D_annotation_simulation, e)

        mask_3D_pred = np.zeros_like(entity_mask_3D)
        for e in entity_masks_3D_pred:
            mask_3D_pred = np.logical_or(mask_3D_pred, e)

        mask_3D_annotation_simulation_sitk = sitk.GetImageFromArray(mask_3D_annotation_simulation.astype('uint8'))
        mask_3D_annotation_simulation_sitk.SetOrigin(mask_3D_sitk.GetOrigin())
        mask_3D_annotation_simulation_sitk.SetSpacing(mask_3D_sitk.GetSpacing())
        mask_3D_annotation_simulation_sitk.SetDirection(mask_3D_sitk.GetDirection())
        output_path = Path(mask_3D_annotation_simulation_path) / f'{patient}.nii.gz'
        sitk.WriteImage(mask_3D_annotation_simulation_sitk, output_path)

        mask_3D_pred_stik = sitk.GetImageFromArray(mask_3D_pred.astype('uint8'))
        mask_3D_pred_stik.SetOrigin(mask_3D_sitk.GetOrigin())
        mask_3D_pred_stik.SetSpacing(mask_3D_sitk.GetSpacing())
        mask_3D_pred_stik.SetDirection(mask_3D_sitk.GetDirection())
        output_path = Path(masks_3D_pred_path) / f'{patient}.nii.gz'
        sitk.WriteImage(mask_3D_pred_stik, output_path)

    torch.cuda.empty_cache()
    del inference_state
    del predictor
    del out_mask_logits, video_segments
    del mask_3D_sitk
    del mask_3D_pred_stik
    del mask_3D_annotation_simulation_sitk
    del labeled_mask_3D
    gc.collect()
    return None
    


# Add argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SAM2 segmentation')
    parser.add_argument('--prompt', type=str, choices=['box', 'mask'], default='box', help='Type of initial prompt: box or mask')
    parser.add_argument('--dice_thresh', type=float, default=0.9, help='Dice threshold value')
    parser.add_argument('--hd_95_thresh', type=float, default=np.inf, help='Hausdorff 95 threshold')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--model_type', type=str, default='sam2_hiera_tiny', help='medsam2 or sam2_hiera_tiny')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    prompt = args.prompt
    dice_thresh = args.dice_thresh
    hd_95_thresh = args.hd_95_thresh
    model_type = args.model_type

    # Run the process loop with the passed args
    warnings = []
    for patient in os.listdir(dataset_path / 'SAM_2_frames_4' / 'imagesTr'):
        torch.cuda.empty_cache()
        warnings.append({'warning': process(model_type, dataset_path, patient, dice_thresh, hd_95_thresh, prompt)})
    pd.DataFrame(warnings).dropna().to_csv(dataset_path / 'SAM_2_frames_4/warnings_video_prediction.csv')

