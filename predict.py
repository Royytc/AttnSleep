import torch
import model.model as module_arch
import numpy as np
# 加载模型

from scipy.signal import resample

model = module_arch.AttnSleep()
checkpoint = torch.load('./saved/Exp1/14_05_2024_10_48_15_fold10/checkpoint-epoch90.pth')
model.load_state_dict(checkpoint["state_dict"])
model.eval()


num_eeg = 4
eeg_data=[[] for i in range(num_eeg)]
with open("./wangjing_23-09_3_TP10_trough_eeg.txt") as file:
    count = 0
    for line in file:
        count += 1
        values = line.split(",")
        if count > 7:
            for i in range(num_eeg):
                eeg_data[i].append(values[i+1])
select_channel = 1
fs=256
start_min = 150
start_index = start_min*fs*60
end_index = int(start_index +30*fs)
eeg_data =np.array(eeg_data)
predict_data = eeg_data[:,start_index:end_index]
predict_data =np.array(predict_data)
predict_data=predict_data.astype(np.float32)
predict_data=np.reshape(predict_data,(4, 1, -1))
predict_data = np.array(list(resample(predict_data[i,0,:],100*30) for i in range(len(predict_data)) ))
predict_data=np.reshape(predict_data,(4, 1, -1))
print(predict_data.shape)
predict_data=torch.from_numpy(predict_data)
output = model(predict_data)
print(output)

# import matplotlib.pyplot as plt

# result=[]
# select_channel = 2
# fs=256
# duration = fs*30
# eeg_data =np.array(eeg_data[select_channel])
# split_data = [eeg_data[i:i+duration] for i in range(0, len(eeg_data), duration)]
# for data in split_data:
#     data = resample(data, 100*30)
#     data = np.reshape(data,(1,1,-1))
#     data = data.astype(np.float32)
#     # print(data.shape)
#     predict_data=torch.from_numpy(data)
#     output = model(predict_data)
#     pred = torch.argmax(output, dim=1)
#     result.append(pred.item())
# print(result)
# # index_list = np.linspace(len(result))
# # index_list = index_list / 2
# plt.figure(figsize=(12,4))
# plt.plot( result)           
# plt.show()

# num_eeg = 4
# result=[[] for i in range(num_eeg)]
# # select_channel = 2

# fs=256
# duration = fs*30
# for i in range(num_eeg):
#     eeg_data =np.array(eeg_data[i])
#     split_data = [eeg_data[i:i+duration] for i in range(0, len(eeg_data), duration)]
#     for data in split_data:
#         data = resample(data, 100*30)
#         data = np.reshape(data,(1,1,-1))
#         data = data.astype(np.float32)
#         # print(data.shape)
#         predict_data=torch.from_numpy(data)
#         output = model(predict_data)
#         # pred = torch.argmax(output, dim=1)
#         result[i].append(output.numpy())
# result = np.array(result)
# pred_result=np.sum([result[0], result[1], result[2], result[3]], axis=0)
# pred_result=[]
# # index_list = np.linspace(len(result))
# # index_list = index_list / 2
# plt.figure(figsize=(12,4))
# plt.plot( result)           
# plt.show()