from typing import List, Tuple, Optional
import torch


class DistillCapture:
    """
    Attaches forward hooks to a torchvision Faster R-CNN model to capture:
    - RPN head outputs: objectness logits and bbox deltas per feature level
    - ROI box predictor outputs: class logits and bbox deltas for proposals
    - ROI proposals per image (to align teacher/student at ROI stage)
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.reset()

    def reset(self):
        self.rpn_objectness: Optional[List[torch.Tensor]] = None
        self.rpn_bbox_deltas: Optional[List[torch.Tensor]] = None
        self.roi_logits: Optional[torch.Tensor] = None
        self.roi_bbox_deltas: Optional[torch.Tensor] = None
        self.roi_proposals: Optional[List[torch.Tensor]] = None

    def attach(self):
        # Hook RPN head: captures (objectness, bbox_regression)
        def rpn_head_hook(module, inputs, output):
            # output is Tuple[List[Tensor], List[Tensor]]
            try:
                obj, reg = output
            except Exception:
                # Some versions return as list; try to parse
                if isinstance(output, (list, tuple)) and len(output) >= 2:
                    obj, reg = output[0], output[1]
                else:
                    return
            self.rpn_objectness = [o.detach() for o in obj]
            self.rpn_bbox_deltas = [r.detach() for r in reg]

        # Hook ROIHeads forward to capture proposals per image
        def roi_heads_hook(module, inputs, output):
            # inputs: (features, proposals, image_shapes, targets)
            if len(inputs) >= 2:
                proposals = inputs[1]
                if isinstance(proposals, (list, tuple)):
                    self.roi_proposals = [p.detach() for p in proposals]

        # Hook box predictor to capture logits and bbox deltas
        def box_pred_hook(module, inputs, output):
            # output: logits, bbox_deltas
            if isinstance(output, (list, tuple)) and len(output) >= 2:
                logits, bbox_deltas = output[0], output[1]
            else:
                logits, bbox_deltas = output, None
            if logits is not None:
                self.roi_logits = logits.detach()
            if bbox_deltas is not None:
                self.roi_bbox_deltas = bbox_deltas.detach()

        # Attach hooks
        if hasattr(self.model, 'rpn') and hasattr(self.model.rpn, 'head'):
            self.hooks.append(self.model.rpn.head.register_forward_hook(rpn_head_hook))
        if hasattr(self.model, 'roi_heads'):
            self.hooks.append(self.model.roi_heads.register_forward_hook(roi_heads_hook))
            if hasattr(self.model.roi_heads, 'box_predictor'):
                self.hooks.append(self.model.roi_heads.box_predictor.register_forward_hook(box_pred_hook))
        return self

    def detach(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def flatten_rpn_objectness(self) -> Optional[torch.Tensor]:
        if not self.rpn_objectness:
            return None
        # list of [N, A, H, W] or [N, A*1, H, W]
        flats = []
        for obj in self.rpn_objectness:
            # Ensure shape [N, A, H, W]
            if obj.dim() == 4:
                N, AHW, H, W = obj.shape
                A = AHW  # if already (A, H, W)
                obj = obj
            else:
                return None
            flats.append(obj.flatten(start_dim=1))  # [N, A*H*W]
        return torch.cat(flats, dim=1)

    def flatten_rpn_bbox(self) -> Optional[torch.Tensor]:
        if not self.rpn_bbox_deltas:
            return None
        flats = []
        for reg in self.rpn_bbox_deltas:
            # reg: [N, A*4, H, W] -> [N, A*H*W, 4]
            if reg.dim() != 4:
                return None
            N, A4, H, W = reg.shape
            reg = reg.view(N, -1, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()  # [N, A, H, W, 4]
            reg = reg.view(N, -1, 4)  # [N, A*H*W, 4]
            flats.append(reg)
        return torch.cat(flats, dim=1)

    def split_roi_outputs_by_image(self) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        if self.roi_logits is None or self.roi_bbox_deltas is None or self.roi_proposals is None:
            return None, None
        counts = [p.shape[0] for p in self.roi_proposals]
        if sum(counts) != self.roi_logits.shape[0]:
            # Unexpected mismatch; bail out gracefully
            return None, None
        logits_list = list(self.roi_logits.split(counts, dim=0))
        bbox_list = list(self.roi_bbox_deltas.split(counts, dim=0))
        return logits_list, bbox_list

