import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# data1 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_neg13.pth")
# data2 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_neg14.pth")
# data3 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos3.pth")
# data4 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos4.pth")
# data5 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos5.pth")
# data6 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos6.pth")
# data7 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos7.pth")
# data8 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos8.pth")
# data9 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos9.pth")
# data10 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos10.pth")
# data11 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos11.pth")
# data12 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos12.pth")
# data13 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos13.pth")
# data14 = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_pos14.pth")
# data = torch.cat((data1,data2),dim=1)
# torch.save(data, "/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/validation_common_representations_neg21.pth")

checkpoint = torch.load("/home/xjg/myTruthX/mytruthx_model_200epoch.pt")
# checkpoint = torch.load("/home/xjg/TruthX/truthx_models/mistral-7b-v0.1/truthx_model.fold1.pt")

# print(checkpoint.keys())
print(checkpoint.items())
# print(checkpoint['state_dict'].keys())
# print(type(checkpoint['pos_center']))
# print(checkpoint['pos_center'].shape)

# def ctr_loss(rep,rep_pos_mean,rep_neg_mean,temperature):
#     logits_pos = torch.nn.functional.cosine_similarity(
#         rep_pos_mean.unsqueeze(0).float(),
#         rep.unsqueeze(1).float(),
#         dim=-1,
#     )
#     logits_neg = torch.nn.functional.cosine_similarity(
#         rep_neg_mean.unsqueeze(0).float(),
#         rep.unsqueeze(1).float(),
#         dim=-1,
#     )
#     logits=torch.cat((logits_pos,logits_neg),dim=-1)
#     logits /= temperature

#     ctr_loss = -torch.nn.LogSoftmax(-1)(logits)
#     ctr_loss = ctr_loss[:,0].mean()
#     return ctr_loss