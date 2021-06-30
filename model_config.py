heatmap_model_config = {"in_channels": 3,
                       "num_classes": 68,
                       "hg_dims": [[256, 256, 384], [384, 384, 512]],
                       "downsample": True
                       }

graph_model_config = {"num_classes": 68,
                      "embedding_hidden_sizes": [32],
                      "class_embedding_size": 1,
                      "edge_hidden_size": 4,
                      # "visual_feature_dim": 1920,     # Stacked Hourglass
                      "visual_feature_dim": 270,        # HRNet
                      "visual_hidden_sizes": [512, 128, 32],
                      "visual_embedding_size": 8,
                      "GCN_dims": [64, 16],
                      "self_connection": False,
                      "graph_norm": "softmax",
                      # "graph_norm": "mean"
                      }
