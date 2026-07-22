import torch
import numpy as np
import time

from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.config import load_config_from_file

from sgg_benchmark.data.build import build_transforms
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.data import get_dataset_statistics

import cv2
import seaborn as sns
import os
import colorsys
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

class SGG_Model(object):
    def __init__(self, config, weights, dcs=100, tracking=False, rel_conf=0.1, box_conf=0.5, show_fps=True) -> None:
        self.cfg = load_config_from_file(config)
        self.cfg.test.custum_eval = True
        self.cfg.output_dir = os.path.dirname(config)

        # to force SGDET mode /!\ careful though, if the model hasn't been trained in sgdet mode, this will break the code
        self.cfg.model.roi_relation_head.use_gt_object_label = False
        self.cfg.model.roi_relation_head.use_gt_box = False

        self.cfg.model.roi_heads.detections_per_img = dcs
        # self.cfg.model.backbone.nms_thresh = 0.267
        self.show_fps = show_fps

        # for visu
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_padding = 2  # Padding around the text

        self.stats = get_dataset_statistics(self.cfg)
        # Do NOT pop __background__: model labels are 1-indexed (0=background),
        # so obj_classes[label] is correct as-is.

        self.obj_class_colors = sns.color_palette('Paired', len(self.stats['obj_classes'])+2)
        # to cv2 format
        self.obj_class_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in self.obj_class_colors]

        logger = setup_logger("sgg_demo")
        logger.remove()
        
        self.model = None
        self.model_weights = weights
        self.checkpointer = None
        self.device = None
        self.tracking = tracking
        self.last_time = 0

        self.rel_conf = rel_conf
        self.box_conf = box_conf

        # can choose between BYTETracker or OCSORT, in my experience OCSORT works a little bit better
        if self.tracking:
            from boxmot import OcSort

            self.tracker = OcSort(
                per_class=True,
                det_thresh=0,
                max_age=20,
                min_hits=1,
                asso_threshold=0.2,
                delta_t=2,
                asso_func='giou',
                inertia=0.2,
                use_byte=True,
            )
            
        self.load_model()
        self.model.roi_heads.eval()
        self.model.eval()

        self.pre_time_bench = []
        self.detec_time_bench = []
        self.post_time_bench = []
        self._last_bboxes = None  # exposed for external heatmap overlays

    def load_model(self):
        self.model = build_detection_model(self.cfg)
        self.model.to(self.cfg.model.device)

        self.checkpointer = DetectronCheckpointer(self.cfg, self.model)
        # last_check = self.checkpointer.get_checkpoint_file()
        # if last_check == "":
        #     last_check = self.cfg.model.weight
        ckpt = self.checkpointer.load(self.model_weights)
        self.device = torch.device(self.cfg.model.device)

        if self.cfg.model.backbone.type == "yolov8world":
            names = self.stats['obj_classes'].values()
            self.model.backbone.load_txt_feats(names)

    def predict(self, image, visu_type='image', return_attention=False):
        self.model.roi_heads.eval()
        self.model.backbone.eval()

        out_img = image.copy()
        self.last_time = time.time()
        img_list, _ = self._pre_processing(image)
        img_list.image_sizes = [(image.shape[0], image.shape[1])]
        img_list = img_list.to(self.device)
        targets = None
        pre_process_time =(time.time()-self.last_time)*1000
        self.pre_time_bench.append(pre_process_time)
        
        with torch.no_grad():
            t_start = time.time()
            predictions = self.model(img_list, targets, return_attention=return_attention)
            det_time = time.time()-t_start # in second
            det_time *= 1000 # in milisecond
            self.detec_time_bench.append(det_time)
        
        if return_attention:
            return predictions[0]

        t_start2 = time.time()
        bboxes, rels = self._post_process2(predictions[0], orig_size=image.shape[:2], box_thres=self.box_conf, rel_threshold=self.rel_conf)
        bboxes = bboxes.cpu().numpy()
        rels = rels.cpu().numpy()
        self._last_bboxes = bboxes  # cache for external consumers (e.g. heatmap demo)
        post_process_time = time.time()
    
        # update tracker
        if self.tracking and len(bboxes) > 0:
            # check if there is bbox to track
            if len(bboxes) > 0:
                tracks = self.tracker.update(bboxes, image)
                # add one dim to bboxes
                bboxes = np.concatenate((bboxes, np.zeros((len(bboxes), 1))), axis=1)
                if len(tracks) > 0:
                    for i, cur_id in enumerate(tracks[:,7]):
                        cur_id = int(cur_id)
                        # update the track id in the predictions
                        class_label = str(int(bboxes[cur_id][5].item()))
                        
                        bboxes[cur_id][6] = int(class_label + str(int(tracks[i][4]))) # we track by class, 140, 141, 142, ...
                        # update the box coordinates
                        bboxes[cur_id][0] = tracks[i][0]
                        bboxes[cur_id][1] = tracks[i][1]
                        bboxes[cur_id][2] = tracks[i][2]
                        bboxes[cur_id][3] = tracks[i][3]
        if visu_type == 'video':
            out_img = self.draw_full_graph(out_img, bboxes, rels)
               # Assuming out_img is the image and predictions is the dictionary containing the predictions
            image_height, image_width = out_img.shape[:2]
            
            # Calculate the font scale based on the image width
            max_text_width = 0.2 * image_width
            font_scale = max_text_width / 250  # Adjust the divisor to fine-tune the font size
            
            # Calculate the height of the text
            (text_width, text_height), baseline = cv2.getTextSize("Sample Text", cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)
            text_height += baseline
            
            # Calculate the positions for the text
            positions = {
                "fps": (10, text_height * 1),
                "objects": (10, image_height - text_height * 1 - 100),
                "relationships": (10, image_height - text_height * 2 - 120),
                "detection": (10, text_height * 2 + 10),
                "pre_process": (10, text_height * 3 + 10),
                "post_process": (10, text_height * 4 + 10)
            }
            
            # Draw the text on the image
            if self.show_fps:
                true_fps = 1/(post_process_time-self.last_time) #({1/(time.time()-self.last_time):.2f})
                cv2.putText(out_img, f"FPS: {true_fps:.2f}", positions["fps"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            
            # cv2.putText(out_img, f"Objects: {len(bboxes)}", positions["objects"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            # cv2.putText(out_img, f"Relationships: {len(rels)}", positions["relationships"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            
            cv2.putText(out_img, f"Detection: {det_time:.2f}ms", positions["detection"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)

            # in milisecond
            post_process_time = post_process_time - t_start2
            post_process_time *= 1000
            self.post_time_bench.append(post_process_time)

            # cv2.putText(out_img, f"Pre process: {pre_process_time:.2f}ms", positions["pre_process"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)

            # cv2.putText(out_img, f"Post process: {post_process_time:.2f}ms", positions["post_process"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)

            return out_img, None
        
        elif visu_type == 'image':
            graph_img = self.visualize_graph(rels, bboxes, image.shape[:2])
            out_img = self.draw_boxes_image(bboxes, out_img)

            return out_img, graph_img
        
        return predictions, None
    
    def draw_boxes_image(self, bboxes, out_img):
        bbox_labels = [self.stats['obj_classes'][int(b[5])] for b in bboxes]
        print(bbox_labels)

        for i, bbox in enumerate(bboxes):
            bbox = [int(b) for b in bbox[:4]]
            label = bbox_labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_padding = 2  # Padding around the text
            
            # Draw bounding box
            cv2.rectangle(out_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            
            text = f"{str(i)}_{label}"
            
            # Calculate text size (width, height) and baseline
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Calculate rectangle coordinates for the background
            rect_start = (bbox[0], bbox[1] - text_height - text_padding )
            rect_end = (bbox[0] + text_width + text_padding * 2, bbox[1] + text_padding )
            
            # Draw background rectangle
            cv2.rectangle(out_img, rect_start, rect_end, (255, 0, 0), cv2.FILLED)
            
            # Draw text
            cv2.putText(out_img, text, (bbox[0] + text_padding, bbox[1] - text_padding ), font, font_scale, (255, 255, 255), font_thickness)
        
        return out_img
    
    def get_color(self, idx):
        """Generate a deterministic color for a class index."""
        # Use golden ratio to space out hues
        h = (idx * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        return (int(b * 255), int(g * 255), int(r * 255))

    def draw_bbox(self, img, bbox, label, cls_id):
        # Convert bbox to integer
        left, top, right, bottom = [int(b) for b in bbox]
        color = self.get_color(cls_id)

        # Nicer box: thicker corners
        cv2.rectangle(img, (left, top), (right, bottom), color, 1)
        length = min(15, int((right - left) * 0.2), int((bottom - top) * 0.2))
        # Top-left corner
        cv2.line(img, (left, top), (left + length, top), color, 3)
        cv2.line(img, (left, top), (left, top + length), color, 3)
        # Top-right corner
        cv2.line(img, (right, top), (right - length, top), color, 3)
        cv2.line(img, (right, top), (right, top + length), color, 3)
        # Bottom-left corner
        cv2.line(img, (left, bottom), (left + length, bottom), color, 3)
        cv2.line(img, (left, bottom), (left, bottom - length), color, 3)
        # Bottom-right corner
        cv2.line(img, (right, bottom), (right - length, bottom), color, 3)
        cv2.line(img, (right, bottom), (right, bottom - length), color, 3)

        # Draw label background at top-left
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        y_text = top - th - 5
        if y_text < 0:
            y_text = top + th + 5
            cv2.rectangle(img, (left, top), (left + tw + 4, top + th + 5), color, -1)
            cv2.putText(img, label, (left + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (left, top - th - 5), (left + tw + 4, top), color, -1)
            cv2.putText(img, label, (left + 2, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Return coordinates of the center of the bbox
        return (left + right) // 2, (top + bottom) // 2

    def draw_full_graph(self, img, bboxes, rels):
        # bboxes is [N, 6+] : x1, y1, x2, y2, score, label, (id)
        if bboxes is None or len(bboxes) == 0:
            return img

        # Precompute class labels and centers
        bbox_labels = [self.stats['obj_classes'][int(b[5])] for b in bboxes]
        centers = []
        for i, bbox in enumerate(bboxes):
            label_text = bbox_labels[i]
            if len(bbox) > 6:
                label_text = f"{int(bbox[6])}_{label_text}"
            
            c = self.draw_bbox(img, bbox[:4], label_text, int(bbox[5]))
            centers.append(c)
        
        # Collect valid relations
        # Sort by rel_score (index 4) if available (output of ONNX export is [s, o, l, triplet, rel_score])
        if rels is None or (hasattr(rels, '__len__') and len(rels) == 0) or (hasattr(rels, 'ndim') and rels.ndim < 2):
            rels_to_draw = []
        elif rels.shape[1] >= 5:
            rels_list = rels.tolist()
            rels_list.sort(key=lambda x: x[3], reverse=True) # Sort by triplet score
            rels_to_draw = rels_list[:12]
        else:
            rels_to_draw = rels.tolist()

        for rel in rels_to_draw:
            s, o, r = int(rel[0]), int(rel[1]), int(rel[2])
            
            p1 = centers[s]
            p2 = centers[o]
            
            # Color and glow
            rel_color = (255, 255, 255) # White line
            glow_color = (255, 128, 0) # Orange glow
            
            # Draw directional arrow from Subject to Object
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist > 40:
                alpha = 20.0 / (dist + 1e-6)
                p1_short = (int(p1[0] * (1-alpha) + p2[0] * alpha), int(p1[1] * (1-alpha) + p2[1] * alpha))
                p2_short = (int(p2[0] * (1-alpha) + p1[0] * alpha), int(p2[1] * (1-alpha) + p1[1] * alpha))
                
                # Draw glow line (thicker)
                cv2.line(img, p1_short, p2_short, glow_color, 2, cv2.LINE_AA)
                cv2.line(img, p1_short, p2_short, rel_color, 1, cv2.LINE_AA)
                
                # Sharp arrowhead
                angle = np.arctan2(p1_short[1] - p2_short[1], p1_short[0] - p2_short[0])
                tip_len = 8
                tip1 = (int(p2_short[0] + tip_len * np.cos(angle + 0.5)), int(p2_short[1] + tip_len * np.sin(angle + 0.5)))
                tip2 = (int(p2_short[0] + tip_len * np.cos(angle - 0.5)), int(p2_short[1] + tip_len * np.sin(angle - 0.5)))
                
                cv2.line(img, p2_short, tip1, rel_color, 1, cv2.LINE_AA)
                cv2.line(img, p2_short, tip2, rel_color, 1, cv2.LINE_AA)
            
            # Relation label
            r_label = self.stats['rel_classes'][r]
            
            # Draw label at 1/3 point from Subject toward Object
            mid = (int(p1[0] * 0.65 + p2[0] * 0.35), int(p1[1] * 0.65 + p2[1] * 0.35))
            
            (tw, th), _ = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (mid[0] - 2, mid[1] - th - 2), (mid[0] + tw + 2, mid[1] + 2), (20, 20, 20), -1)
            cv2.putText(img, r_label, (mid[0], mid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return img
    
    def draw_rounded_rectangle(self, img, top_left, bottom_right, color, thickness, radius):
        # Draw the four straight edges
        cv2.rectangle(img, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]), color, thickness)
        cv2.rectangle(img, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color, thickness)

        # Draw the four rounded corners
        cv2.ellipse(img, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 90, 0, 90, color, thickness)

    def visualize_graph(self, rels, bboxes, color='blue'):
        bbox_labels = [self.stats['obj_classes'][int(b[5])] for b in bboxes]
        G = nx.MultiDiGraph()
        for i, r_label in enumerate(rels[:, 2]):
            label_rel = self.stats['rel_classes'][int(r_label)]
            r = rels[i]
            # to int
            r = r.astype(int)
            subj = str(r[0])+'_'+bbox_labels[int(r[0])]
            obj = str(r[1])+'_'+bbox_labels[int(r[1])]
            G.add_edge(str(subj), str(obj), label=label_rel, color=color)

        # draw networkx graph with graphviz, display edge labels
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        G.graph['graph'] = {'scale': '2'}
        G.graph['node'] = {'shape': 'rectangle'}
        # all graph color to blue
        G.graph['edge']['color'] = color
        G.graph['node']['color'] = color

        img_graph = to_agraph(G)
        # Layout the graph
        img_graph.layout('dot')

        # Draw the graph directly to a byte array
        png_byte_array = img_graph.draw(format='png', prog='dot')

        # Convert the byte array to an OpenCV image without redundant conversion
        img_cv2 = cv2.imdecode(np.frombuffer(png_byte_array, np.uint8), cv2.IMREAD_COLOR)

        return img_cv2
    
    def nice_plot(self, img, graph):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)

        out_img = np.zeros((img.shape[0]+graph.shape[0], max(img.shape[1], graph.shape[1]), 3), dtype=np.uint8)
        out_img[:img.shape[0], :img.shape[1]] = img
        out_img[img.shape[0]:, :graph.shape[1]] = graph
        
        return out_img


    def _pre_processing(self, image):
        # to cv2 format
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = torch.LongTensor([-1])
        transform = build_transforms(self.cfg, is_train=False)

        image, target = transform(image, target)
        # image = image[None,:] # add batch dimension
        image_list = to_image_list(image, self.cfg.dataloader.size_divisibility)

        return image_list, target
    
    def _post_process(self, box_dict, rel_threshold=0.1, box_thres=0.1, orig_size=(640,640)):
        height, width = orig_size
        
        # Scaling boxes
        old_width, old_height = box_dict['image_size']
        scale_x = width / old_width
        scale_y = height / old_height
        
        xyxy_bbox = box_dict['boxes'].clone()
        xyxy_bbox[:, [0, 2]] *= scale_x
        xyxy_bbox[:, [1, 3]] *= scale_y

        # current sgg info
        current_dict = {'bbox': [], 
                        'bbox_labels': [], 
                        'bbox_scores': [], 
                        'rel_pairs': [], 
                        'rel_labels': [], 
                        'rel_scores': [], 
                        'rel_all_scores': []
        }
        
        # sort boxes based on confidence
        sortedid, id2sorted = self.get_sorted_bbox_mapping(box_dict['pred_scores'].tolist())
        # filter by box thres
        sortedid = [i for i in sortedid if box_dict['pred_scores'][i] > box_thres]
        id2sorted = {v: k for k, v in enumerate(sortedid)}

        for i in sortedid:
            current_dict['bbox'].append([int(round(b)) for b in xyxy_bbox[i].tolist()])
            current_dict['bbox_labels'].append(box_dict['pred_labels'][i].item())
            current_dict['bbox_scores'].append(box_dict['pred_scores'][i].item())

        current_dict['bbox_labels'] = [c for c in current_dict['bbox_labels']]
        
        # transform bbox, bbox_labels and bbox_scores to a single tensor of shape (N, 6)
        bboxes_tensor = torch.cat([torch.tensor(current_dict['bbox']), torch.tensor(current_dict['bbox_scores']).unsqueeze(1), torch.arange(len(current_dict['bbox_labels'])).unsqueeze(1)], dim=1)

        # sorted relationships
        rel_sortedid, _ = self.get_sorted_bbox_mapping(box_dict['pred_rel_scores'][:,1:].max(1)[0].tolist())

        # remove all relationships with score < rel_threshold
        rel_sortedid = [i for i in rel_sortedid if box_dict['pred_rel_scores'][i][1:].max(0)[0] > rel_threshold]

        # sorted rel
        for i in rel_sortedid:
            old_pair = box_dict['rel_pair_idxs'][i].tolist()
            # don't add if the subject or object is not in the sortedid
            if old_pair[0] not in id2sorted or old_pair[1] not in id2sorted:
                continue
            
            current_dict['rel_labels'].append(box_dict['pred_rel_scores'][i][1:].max(0)[1].item() + 1)
            rel_s = box_dict['pred_rel_scores'][i][1:].max(0)[0].item()
            # rel score is sub_score * obj_score * rel_score
            rel_s = rel_s * box_dict['pred_scores'][old_pair[0]].item() * box_dict['pred_scores'][old_pair[1]].item()
            current_dict['rel_scores'].append(rel_s)
            current_dict['rel_pairs'].append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['bbox_labels'] = [self.stats['obj_classes'][i] for i in current_dict['bbox_labels']]
        current_dict['rel_labels'] = [self.stats['rel_classes'][i] for i in current_dict['rel_labels']]

        current_dict['yolo_bboxes'] = bboxes_tensor

        return current_dict
    
    def get_sorted_bbox_mapping(self, score_list):
        sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
        sorted2id = [item[1] for item in sorted_scoreidx]
        id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
        return sorted2id, id2sorted
    
    def _post_process2(self, box_dict, rel_threshold=0.1, box_thres=0.1, orig_size=(640,640)):
        height, width = orig_size

        # Scaling boxes
        old_width, old_height = box_dict['image_size']
        scale_x = width / old_width
        scale_y = height / old_height

        xyxy_bbox = box_dict['boxes'].clone()
        xyxy_bbox[:, [0, 2]] *= scale_x
        xyxy_bbox[:, [1, 3]] *= scale_y

        bbox_scores = box_dict['pred_scores']
        bbox_labels = box_dict['pred_labels']

        # Filter boxes by score — independently of whether they appear in any relation.
        # This ensures boxes are always shown even when no relation survives.
        filtered_bbox_ids = torch.where(bbox_scores > box_thres)[0]

        if filtered_bbox_ids.numel() == 0:
            return torch.tensor([]), torch.tensor([])

        # Build output boxes: [x1, y1, x2, y2, score, label]
        bboxes_tensor = torch.cat([
            xyxy_bbox[filtered_bbox_ids].int().float(),
            bbox_scores[filtered_bbox_ids].unsqueeze(1),
            bbox_labels[filtered_bbox_ids].unsqueeze(1).float(),
        ], dim=1)

        # --- Relations ---
        rel_scores_full = box_dict['pred_rel_scores']  # (#rel, #rel_class) softmax
        pairs = box_dict['rel_pair_idxs']              # (#rel, 2)
        rel_labels = box_dict['pred_rel_labels']       # (#rel,)

        if pairs.shape[0] == 0:
            return bboxes_tensor, torch.tensor([])

        # Geometric mean: (rel × subj × obj)^(1/3) — keeps the score on the same [0,1] scale
        # as each individual component, so rel_conf=0.1 is a meaningful threshold.
        # Raw product would give e.g. 0.3^3 = 0.027, forcing users to use rel_conf=0.001.
        fg_rel_scores = rel_scores_full[:, 1:].max(dim=1)[0]
        triplet_scores = (fg_rel_scores * bbox_scores[pairs[:, 0]] * bbox_scores[pairs[:, 1]]) ** (1.0 / 3.0)

        # Stack: [subj_idx, obj_idx, rel_label, triplet_score]
        all_rels = torch.cat([
            pairs.int(),
            rel_labels.unsqueeze(1).int(),
            triplet_scores.unsqueeze(1),
        ], dim=1)

        # Only keep rels where both endpoints survived box_thres
        fids = filtered_bbox_ids.unsqueeze(0)           # (1, N)
        subj_ok = (all_rels[:, 0].unsqueeze(1) == fids).any(dim=1)
        obj_ok  = (all_rels[:, 1].unsqueeze(1) == fids).any(dim=1)
        all_rels = all_rels[subj_ok & obj_ok]

        # Filter by triplet threshold
        all_rels = all_rels[all_rels[:, 3] > rel_threshold]

        if all_rels.size(0) == 0:
            return bboxes_tensor, torch.tensor([])

        # Remap subj/obj indices from original-box space to filtered-box space
        max_orig_idx = int(filtered_bbox_ids.max().item()) + 1
        idx_map = torch.full((max_orig_idx,), -1, dtype=torch.long, device=filtered_bbox_ids.device)
        idx_map[filtered_bbox_ids] = torch.arange(len(filtered_bbox_ids), device=filtered_bbox_ids.device)

        all_rels[:, 0] = idx_map[all_rels[:, 0].long()]
        all_rels[:, 1] = idx_map[all_rels[:, 1].long()]

        return bboxes_tensor, all_rels
    
    def post_process_rels(self, box_dict):
        rel_scores = box_dict['pred_rel_scores']
        pairs = box_dict['rel_pair_idxs']
        labels = box_dict['pred_rel_labels']
        
        # Remove the background and get the max scores
        rel_scores = rel_scores[:, 1:].max(dim=1)[0]

        # Stack the pairs, labels, and rel_scores
        all_rels = torch.cat((pairs.int(), labels.unsqueeze(1).int(), rel_scores.unsqueeze(1)), dim=1)
        
        # Remove all entries with subj == obj
        #all_rels = all_rels[all_rels[:, 0] != all_rels[:, 1]]
        
        # Check if there are relations
        if all_rels.size(0) == 0:
            return all_rels

        return all_rels
    
    def get_latency(self):
        print(f"Preprocessing time: {np.mean(self.pre_time_bench):.2f} ms")
        print(f"Detection time: {np.mean(self.detec_time_bench):.2f} ms")
        print(f"Post processing time: {np.mean(self.post_time_bench):.2f} ms")
        print(f"Total time: {np.mean(self.pre_time_bench) + np.mean(self.detec_time_bench) + np.mean(self.post_time_bench):.2f} ms")