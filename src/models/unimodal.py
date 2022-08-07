import os
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

from utils import preprocessing

# Resnet-152 layer res-5c (Image-grid) with BERT
class ResBERT(nn.Module):
    def __init__(self, train, test, **kwargs) -> None:
        super().__init__()

        #START############################ VARIABLES #########################

        self.lr = kwargs.pop('lr', 1e-1)
        self.lr_decay = kwargs.pop('lr_decay', 1e-4)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.accumulation = kwargs.pop('accumulation', 4)
        self.num_epochs = kwargs.pop('num_epochs', 90)
        self.start_epoch = kwargs.pop('start_epoch', 0) #'manual epoch number (useful on restarts)'
        self.momentum = kwargs.pop('momentum', 0.9)
        self.seed = kwargs.pop('seed', 42)
        self.workers = kwargs.pop('workers', 4)

        self.resume = kwargs.pop('resume', None) # path to checkpoints folder containing checkpoints for tokenizer, language model and image model
        self.feature_extracting = kwargs.pop('feature_extracting', False)
        self.prefix = kwargs.pop('prefix', None) # prefix to add to save file path

        self.top_k = kwargs.pop('top_k', 3)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.print_model = kwargs.pop('print_model', False)
        self.save_model = kwargs.pop('save_model', True)

        #END########################################################################

        # handling gpu
        if torch.cuda.is_available():
            print('CUDA available')
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            
        else:
            print('WARNING: CUDA is not available')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #START########################## INIT MODEL ###############################################

        # Download model
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        self.gpt2 = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')    # Download model and configuration from huggingface.co and cache.
        
        # Load from checkpoint file
        if self.resume:
            if os.path.isdir(self.resume):
                print(f"=> loading checkpoints from directory '{self.resume}'")
                gpt2_checkpoint = torch.load(os.path.join(self.resume, 'ResGPT2_gpt2'))
                tokenizer_checkpoint = torch.load(os.path.join(self.resume, 'ResGPT2_tokenizer'))
                resnet_checkpoint = torch.load(os.path.join(self.resume, 'ResGPT2_resnet'))
            
                self.start_epoch = resnet_checkpoint['epoch']
                self.best_acc1 = resnet_checkpoint['best_acc1']
                
                self.model.load_state_dict(resnet_checkpoint['state_dict'])
                self.optimizer.load_state_dict(resnet_checkpoint['optimizer'])
                self.scheduler.load_state_dict(resnet_checkpoint['scheduler'])
                print(f"=> loaded checkpoint '{self.resume}' (epoch {resnet_checkpoint['epoch']})")

        else:
            print(f"=> Given path is not a directory '{self.resume}'")

        # Print Model
        print(f'Created model: {self.model_name}')
            model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2', output_attentions=True)  # Update configuration during loading

        if self.feature_extracting:
            self._set_parameter_requires_grad(self.gpt2, self.feature_extracting)
            self._set_parameter_requires_grad(self.resnet, self.feature_extracting)


        self.resnet.to(self.device)
        self.gpt2.to(self.device)

        if self.print_model:
            print(self.model.eval())

        # initialize criterion, optimizer and scheduler

        #END############################################################################################################

        