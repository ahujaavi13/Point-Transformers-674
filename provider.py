import torch


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = torch.rand((B, 3))*(2*shift_range)-shift_range
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data
    
def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = torch.rand(B)*(scale_high-scale_low)+scale_low
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
      dropout_ratio = torch.rand(1)[0] * max_dropout_ratio
      drop_idx = torch.where(torch.rand(batch_pc.shape[1]) <= dropout_ratio)[0]
      if len(drop_idx) > 0:
          batch_pc[b, drop_idx, :] = batch_pc[b, 0, :].clone()
    return batch_pc
