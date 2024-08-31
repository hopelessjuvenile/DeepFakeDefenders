import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from einops import rearrange
# from torchinfo import summary
import torch
import torch.nn.functional as F
import math


class FFDV_Cov(nn.Module):
    def __init__(self, num_class, extra_model_path, temporal, device, mode="train"):
        super(FFDV_Cov, self).__init__()
        self.fc_dim = 0
        self.num_class = num_class
        self.extra_model_path = extra_model_path
        self.temporal = temporal
        self.device = device
        self.mode = mode
        self.feature_extractor = self.get_models(extra_model_path)
        
        self.alpha = 0.01
        self.num_slots = 2000
        self.shrink_thres = 0.0005
        
        self.audio_fc = nn.Linear(512, self.temporal)
        self.cov_fc = nn.Sequential(nn.Linear(self.temporal, self.temporal), 
                                    nn.ReLU())
        self.cls_head = nn.Linear(self.temporal * 2, num_class)
        self.softmax = nn.Softmax()

    def get_models(self, path):
        resnet18 = models.resnet18()
        self.fc_dim = resnet18.fc.in_features
        resnet18_feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
        if self.mode == "train":
            resnet18_feature_extractor.load_state_dict(torch.load(path))
        return resnet18_feature_extractor

    def min_max_norm(self, x):
        result_max = x.max()
        result_min = x.min()
        x = (x - result_min) / (result_max - result_min)
        return x
    
    def forward(self, videos, audios):
        batch, temporal = videos.shape[0], videos.shape[1]

        audio_features = self.feature_extractor(audios)
        videos_sum = torch.sum(videos, dim=2, dtype=torch.float)
        audio_features = audio_features.squeeze(3).squeeze(2)

        video_features = self.cov_fc(videos_sum) + videos_sum * self.alpha
        audio_features = self.audio_fc(audio_features)

        cosine_similarities = []
        for i in range(video_features.shape[0]):
            video_feature_i = video_features[i, :]
            audio_feature_i = audio_features[i, :]

            video_feature_i = video_feature_i / video_feature_i.norm(dim=0)
            audio_feature_i = audio_feature_i / audio_feature_i.norm(dim=0)

            cosine_similarity_i = torch.dot(video_feature_i, audio_feature_i)
            cosine_similarities.append(cosine_similarity_i)
        
        cosine_similarities = torch.tensor(cosine_similarities, dtype=video_features.dtype).to(video_features.device)
        cosine_similarities = cosine_similarities.unsqueeze(1)

        all_features = torch.cat((video_features, audio_features), dim=1)

        result = self.cls_head(all_features)
        # result_max = result[:, 1].max()
        # result_min = result[:, 1].min()
        # result[:, 1] = (result[:, 1] - result_min) / (result_max - result_min)
        # print(cosine_similarities)
        temp = result

        if self.mode == "train":
            temp = temp + cosine_similarities
        if self.mode == "test" or self.mode == "val":
            temp[:, 1] = self.min_max_norm(temp[:, 1])
            temp[:, 1] = temp[:, 1] + cosine_similarities[:, 0] * 60
        if self.mode == "inference":
            temp = self.softmax(temp)
            temp[:, 1] = temp[:, 1] + cosine_similarities[:, 0] * 60
        result = temp
        return result


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # video_f = torch.randn((2, 34, 3, 256, 256), dtype=torch.float32)
    # audio_f = torch.randn((2, 3, 256, 256), dtype=torch.float32)
    video_f = torch.randn((2, 128, 128), dtype=torch.float32)
    audio_f = torch.randn((2, 3, 256, 256), dtype=torch.float32)
    raw_data = {"video": video_f, "audio": audio_f}
    raw_data["video"], raw_data["audio"] = raw_data["video"].to(device), raw_data["audio"].to(device)
    model = FFDV_Cov(num_class=2, extra_model_path="pretrain_weight/resnet18_no_linear.pth", temporal=128, device=device).cuda(device)
    torch.save(model.state_dict(), "FFDV_Cov_test.pth")
    # summary(model, input_data=raw_data)
    # output = model(raw_data)
    # print(output.shape)
    # print(audio.shape)