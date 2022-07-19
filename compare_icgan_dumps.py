import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(1, "/private/home/vkhalidov/2022_icgan/ic_gan")

ICGAN_BASEDIR = "/checkpoint/vkhalidov/2022_icgan/debug_icgan/"
ICGAN_VK_BASEDIR = "/checkpoint/vkhalidov/2022_icgan/debug_icgan_vkhalidov/"

def is_close(v1, v2):
    if isinstance(v1, torch.Tensor) and isinstance(v2, np.ndarray):
        v1 = v1.cpu().numpy()
    if type(v1) != type(v2):
        return False, f"not the same type: type1={type(v1)}, type2={type(v2)}"
    if isinstance(v1, tuple):
        if len(v1) != len(v2):
            return False, f"tuple len1={len(v1)}, len2={len(v2)}"
        for i in range(len(v1)):
            res_i, detail_i = is_close(v1[i], v2[i])
            if not res_i:
                return False, f"tuple element {i}: {detail_i}"
        return True, ""
    elif isinstance(v1, torch.Tensor):
        print(v1.shape)
        if v1.shape != v2.shape:
            return False, f"tensor shape1={v1.shape}, shape2={v2.shape}"
        l1_dist = torch.sum(torch.abs(v1 - v2))
        if l1_dist > 1e-5:
            return False, f"tensor diff {l1_dist}\n{v1}\n{v2}"
        return True, ""
    elif isinstance(v1, np.ndarray):
        if v1.shape != v2.shape:
            return False, f"ndarray shape1={v1.shape}, shape2={v2.shape}"
        l1_dist = np.sum(np.absolute(v1 - v2))
        if l1_dist > 1e-5:
            return False, f"ndarray diff {l1_dist}\n{v1}\n{v2}"
        return True, ""
    elif v1 is None:
        if v2 is not None:
            return False, f"v1={v1}, v2={v2}"
        return True, ""
    return False, f"Not implemented for {type(v1)} and {type(v2)}"


def verify_sample_conditionings():
    for it in range(50):
        print(f"processing sample conditionings {it}")
        sam_cond_fpath = os.path.join(ICGAN_BASEDIR, f"sample_conditionings_{it:04d}.dat")
        sam_cond_vk_fpath = os.path.join(ICGAN_VK_BASEDIR, f"sample_conditionings_{it:04d}.dat")
        with open(sam_cond_fpath, "rb") as sam_cond_f:
            sam_cond = pickle.load(sam_cond_f)
        with open(sam_cond_vk_fpath, "rb") as sam_cond_vk_f:
            sam_cond_vk = pickle.load(sam_cond_vk_f)
        for key in ["sel_idxs", "instance_gen", "labels_gen"]:
            res, detail = is_close(sam_cond[key], sam_cond_vk[key])
            assert res, f"{key}: not close: {detail}"


def verify_dstep():
    for it in range(5):
        for acc_it in range(32):
            dstep_acc_data_fpath = os.path.join(ICGAN_BASEDIR, f"train_Dstep_{it:04d}_{acc_it:02d}.dat")
            dstep_vk_acc_data_fpath = os.path.join(ICGAN_VK_BASEDIR, f"train_Dstep_{it:04d}_{acc_it:02d}.dat")
            with open(dstep_acc_data_fpath, "rb") as acc_data_f:
                acc_data = pickle.load(acc_data_f)
            with open(dstep_vk_acc_data_fpath, "rb") as vk_acc_data_f:
                vk_acc_data = pickle.load(vk_acc_data_f)
            for key in ["x", "y", "z", "labels", "features"]:
                print(f"processing Dstep {it} {acc_it} {key}")
                #assert type(acc_data[key]) == type(vk_acc_data[key]), f"{key}: acc data {type(acc_data[key])}, vk acc data {type(vk_acc_data[key])}"
                res, detail = is_close(acc_data[key], vk_acc_data[key])
                if not res:
                    print(f"{key}: not close: {detail}")
                assert res, f"{key}: not close: {detail}"

def verify_gstep():
    for it in range(5):
        for acc_it in range(32):
            dstep_acc_data_fpath = os.path.join(ICGAN_BASEDIR, f"train_Gstep_{it:04d}_{acc_it:02d}.dat")
            dstep_vk_acc_data_fpath = os.path.join(ICGAN_VK_BASEDIR, f"train_Gstep_{it:04d}_{acc_it:02d}.dat")
            with open(dstep_acc_data_fpath, "rb") as acc_data_f:
                acc_data = pickle.load(acc_data_f)
            with open(dstep_vk_acc_data_fpath, "rb") as vk_acc_data_f:
                vk_acc_data = pickle.load(vk_acc_data_f)
            for key in ["x", "y", "z", "labels", "features"]:
                print(f"processing Gstep {it} {acc_it} {key}")
                #assert type(acc_data[key]) == type(vk_acc_data[key]), f"{key}: acc data {type(acc_data[key])}, vk acc data {type(vk_acc_data[key])}"
                res, detail = is_close(acc_data[key], vk_acc_data[key])
                if not res:
                    print(f"{key}: not close: {detail}")
                assert res, f"{key}: not close: {detail}"


def main():
    verify_sample_conditionings()
    verify_dstep()
    verify_gstep()

if __name__ == "__main__":
    main()
