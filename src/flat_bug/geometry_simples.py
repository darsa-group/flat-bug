import torch

def find_contours(neighbors):
    _device = neighbors[0].device
    outer_idx = torch.tensor([i for i, n in enumerate(neighbors) if len(n) < 9], dtype=torch.long, device=_device)
    inner_idx = torch.tensor([i for i in range(len(neighbors)) if i not in outer_idx], dtype=torch.long, device=_device)
    outer_points = [neighbors[i][torch.isin(neighbors[i], outer_idx)] for i in outer_idx]
    # Remap indices for outer points
    outer_remap = torch.arange(len(neighbors), dtype=torch.long, device=_device)
    outer_remap[outer_idx] = torch.arange(len(outer_idx), dtype=torch.long, device=_device)
    outer_remap[inner_idx] = -1
    outer_points = [outer_remap[o] for o in outer_points]
    
    skippers = torch.zeros(len(neighbors), dtype=torch.bool, device=_device)
    winners = skippers.clone()

    group_ind = 0
    while group_ind < len(outer_points):
        if skippers[group_ind]:
            group_ind += 1
            continue
        last_added = outer_points[group_ind]
        skippers[group_ind] = True
        winners[group_ind] = True
        while True:
            this = outer_points[group_ind]
            
            mergers = torch.zeros(len(outer_points), dtype=torch.bool, device=_device)
            new_neighbors = mergers.clone()
            for i, o in enumerate(outer_points):
                if skippers[i]:
                    continue
                old_neighbors = torch.isin(o, last_added, assume_unique=True)
                if not old_neighbors.any():
                    continue
                mergers[i] = True
                skippers[i] = True
                if old_neighbors.all():
                    continue
                new_neighbors[o[~old_neighbors]] = True

            if not mergers.any():
                break
            
            last_added = torch.where(new_neighbors)[0].unique()
            outer_points[group_ind] = torch.cat([last_added, this])
        group_ind += 1
    return [outer_idx[n] for n, w in zip(outer_points, winners) if w], inner_idx

def find_neighbors(mask, pos):
    raise NotImplementedError("Seems to be some bug with this function, but I cannot reproduce it at the moment. Only happens on real data, as far as I have been able to find.")
    nmask = torch.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=torch.long, device=mask.device)
    nmask[*(pos + 1).T] = torch.arange(len(pos), device=mask.device, dtype=torch.long) + 1
    nidx = torch.arange(3, device=mask.device, dtype=torch.long).unsqueeze(0).repeat(len(pos), 3) - 1
    nidx += (pos[:, 0].unsqueeze(1) + 1) + (pos[:, 1].unsqueeze(1) + 1) * nmask.shape[0]
    nidx[:, :3] -= nmask.shape[0]
    nidx[:, -3:] += nmask.shape[0]
    # assert (nmask.flatten()[nidx[:, 4]].sort().values == torch.arange(len(pos), device=mask.device, dtype=torch.long) + 1).all(), f"Centers {nidx[:, 4].sort().values} do not match {nmask.flatten().nonzero(as_tuple=False).flatten().sort().values}"

    return [neighbors[neighbors != 0] - 1 for neighbors in nmask.flatten()[nidx]]

def find_neighbors_naive(mask):
    pos = mask.nonzero()
    return [torch.where(((pos[i].unsqueeze(0) - pos) ** 2).sum(dim=1).sqrt() < 1.5)[0] for i in range(len(pos))]
    

def find_contigs(mask):
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise NotImplementedError("Only implemented for 2D masks")
    # start = time.time()

    ## Find the points in the mask
    pos = mask.nonzero(as_tuple=False)
    # neighbors = find_neighbors(mask, pos) # This does not work at the moment for some reason, but could perhaps be faster than the naive version
    
    ## For every point, find the neighbors
    neighbors = find_neighbors_naive(mask)
    
    # neighbor_finding_time = time.time() - start
    # start = time.time()

    ## Find the contigous contours and the inner points
    contours, inners = find_contours(neighbors)
    
    # contour_finding_time = time.time() - start
    # start = time.time()

    ## Convert to float for distance calculation
    pos = pos.float()
    contours = [pos[c] for c in contours]
    ## Assign inner points to contours
    inner_to_contour_min_dist = [(torch.cdist(pos[inners], c)).min(dim=1).values for c in contours]
    inner_to_contour_min_dist = torch.stack(inner_to_contour_min_dist)
    which_contour = inner_to_contour_min_dist.argmin(dim=0)
    ## Combine contours and inner points
    for i, c in enumerate(contours):
        c = torch.cat([c, pos[inners[which_contour == i]]])
        contours[i] = c.long()

    # inner_assigment_time = time.time() - start
    # start = time.time()
        
    ## Initialize the disjoint masks
    split_masks = torch.zeros((len(contours), *mask.shape), dtype=torch.bool, device=mask.device)
    ## Fill the contigous masks into separate disjoint masks
    for i, c in enumerate(contours):
        split_masks[i, c[:,0], c[:,1]] = True

    # mask_creation_time = time.time() - start
    # total_time = neighbor_finding_time + contour_finding_time + inner_assigment_time + mask_creation_time
    # print(f'Found {len(split_masks)} in {total_time:.2f} seconds | Neighbors {neighbor_finding_time:.2f} ({neighbor_finding_time/total_time*100:.3g}%) | Contours {contour_finding_time:.2f} ({contour_finding_time/total_time*100:.3g}%) | Inner Assignment {inner_assigment_time:.3f} ({inner_assigment_time/total_time*100:.3g}%) | Mask Creation {mask_creation_time:.3f} ({mask_creation_time/total_time*100:.3g}%)')
    
    return split_masks

def expand_mask(mask, n=1, dtype=torch.float16):
    neighbor_kernel = torch.ones(1, 1, 1+2*n, 1+2*n, device=mask.device, dtype=dtype)
    return torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0).to(dtype=dtype, device=mask.device), neighbor_kernel, padding=n).squeeze(0).squeeze(0) > 0.5