import torch.nn as nn
import logging
logging.basicConfig(level="INFO")

import vocabulary as vocab


class AEJEPSMSELoss(nn.Module):
    """
    A loss function that implements a custom loss whose equation is shown below.

    .. math::
        L = (\hat{I} - I)^2 + \sum_{i=1}^{m} {(\hat{W_i} - W_i)^2} + \sum_{i=1}^{n} {(\hat{M_i} - M_i)^2} + R(h)

    where \(\hat{I}\), \(\hat{M_i}\) and \(\hat{W_i}\) are the reconstructed image, motor command at time step i and
    reconstructed word at time step i respectively, I, \(M_i\) and \(W_i\) are the target outputs at time step i respectively,
    h is the hidden representation and R(h) is a regularization function that ensures the network learns useful representation.

    Parameters
    ----------
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed.
        Default: 'mean'
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, goal_img_out, text_out, cmd_out, goal_img, text_target, cmd_target):
        """

        Parameters
        ----------
        goal_img_out : torch.Tensor
            The goal image output from the model
        text_out : torch.Tensor
            The text output from the model
        cmd_out : torch.Tensor
            The motor command output from the model
        goal_img : torch.Tensor
            The reference goal image
        text_target : torch.Tensor
            The reference text sequence
        cmd_target : torch.Tensor
            The reference motor command sequence

        Returns
        -------
        The computed loss
        """

        # try Structural Similarity
        L_img = nn.functional.mse_loss(
            goal_img_out, goal_img, reduction=self.reduction)

        # change to WER or PERPLEXITY
        L_text = nn.functional.cross_entropy(
            text_out.float(), 
            text_target.float(), 
            reduction=self.reduction,
            ignore_index = 47
        )

        # change to WER or PERPLEXITY
        L_cmd = nn.functional.mse_loss(
            cmd_out.float(),
            cmd_target.float(),
            reduction=self.reduction
        )

        return L_img, L_text, L_cmd


class AEJEPSCrossEntropyLoss(nn.Module):
    """
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        per_img_out,
        goal_img_out,
        text_out,
        cmd_out,
        per_img,
        goal_img,
        text_target,
        cmd_target,
        ignore_idx:int=0,
        debug: bool = False
    ):
        """
        """

        # try Structural Similarity
        L_img_per = nn.functional.mse_loss(
            per_img_out,
            per_img,
            reduction="mean"
        )
        L_img_goal = nn.functional.mse_loss(
            goal_img_out,
            goal_img,
            reduction="mean"
        )

        L_img = (0.5) * (L_img_per + L_img_goal)
        if debug:
            print(f"\nImg loss: {L_img}")

        # change to WER or PERPLEXITY
        # print(text_out.dtype, text_target.dtype)
        # print("text_out[0].shape: ", text_out[0].shape)
        # print("text_target[0].shape: ", text_target[0].shape)

        L_text = nn.functional.cross_entropy(
            input=text_out,
            target=text_target,
            ignore_index=ignore_idx
        )
        if debug:
            print(f"\nText loss: {L_text}")

        # change to WER or PERPLEXITY
        # print(cmd_out.dtype, cmd_target.dtype)
        # print("cmd_out[0]: ", cmd_out[0])
        # print("cmd_target[0]: ", cmd_target[0])
        L_cmd = nn.functional.cross_entropy(
            cmd_out,
            cmd_target,
            ignore_index=ignore_idx
        )

        if debug:
            print(f"\nCmd loss: {L_cmd}")

        return L_img, L_text, L_cmd


LOSSES = {
    "cross_entropy": AEJEPSCrossEntropyLoss,
    "bce": nn.BCELoss,
    "mse": nn.MSELoss,
    'smooth_l1': nn.SmoothL1Loss,
    "aejeps_loss": AEJEPSMSELoss,
}


def get_loss_func(loss_name):
    """
    Returns a class for a loss function that can be instantiated.
    Parameters
    ----------
    loss_name : str
        The name of the loss function to use can be one cross_entropy | bce | mse | smooth_l1 | aejeps_loss

    Returns
    -------
    A torch.nn.Module subclass implementing the loss specified.
    """

    if loss_name not in LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))

    logging.info(f"Objective: {loss_name}\n")

    return LOSSES[loss_name]
