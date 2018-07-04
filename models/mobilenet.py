import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ["mobilenet", "mobilenetr", "mobileneta", "mobilenetra", "mobilenetg4", "mobilenetrg4", "mobilenetag4", "mobilenetrag4"]

class DepthwiseSepBlock(nn.Module):
	def __init__(self, inplanes, outplanes, stride, useResidual, useAttention, groupScale = 1):
		super(DepthwiseSepBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size = 3, padding = 1, bias = False, stride = stride, groups = inplanes//groupScale)
		self.bn1 = nn.BatchNorm2d(outplanes)
		self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(outplanes)
		self.relu = nn.ReLU(inplace = True)
		self.useResidual = useResidual
		self.useAttention = useAttention
		self.convResidual = None


		if useResidual and ((stride != 1) or (inplanes != outplanes)):
			self.convResidual = nn.Sequential(
				nn.Conv2d(inplanes, outplanes, kernel_size = 1, stride = stride, bias = False),
				nn.BatchNorm2d(outplanes)
			)
		#end if
		self.convAttention = None
		if useAttention:
			self.convAttention = nn.Sequential(
				nn.Conv2d(outplanes, outplanes//4, kernel_size = 1, bias = False),
				nn.ReLU(inplace = True),
				nn.Conv2d(outplanes//4, outplanes, kernel_size = 1, bias = False),
				nn.Sigmoid()
			)
		#end if
	#end __init__

	def forward(self, x):
		xClamp = x
		residual = xClamp
		out = self.relu(self.bn1(self.conv1(xClamp)))
		out = self.bn2(self.conv2(out))
		if self.convAttention:
			outC = F.avg_pool2d(out, out.size(2))
			outC = self.convAttention(outC)
			out = out * outC
		#end if
		if self.useResidual:
			if self.convResidual:
				residual = self.convResidual(xClamp)
			#end if
			out = out + residual
		#end if
		out = self.relu(out)
		return out
	#end for
#end


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
#end conv_dw

class MobileNet(nn.Module):
	def __init__(self, useResidual, useAttention, groupScale = 1):
		super(MobileNet, self).__init__()
	
		def conv_bn(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True))

		self.model = nn.Sequential(
			conv_bn(  3,  32, 2), 
			DepthwiseSepBlock( 32,  64, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock( 64, 128, 2, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(128, 128, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(128, 256, 2, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(256, 256, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(256, 512, 2, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(512, 512, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(512, 512, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(512, 512, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(512, 512, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(512, 512, 1, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(512, 1024, 2, useResidual, useAttention, groupScale),
			DepthwiseSepBlock(1024, 1024, 1, useResidual, useAttention, groupScale),
			nn.AvgPool2d(7),
			)
		self.fc = nn.Linear(1024, 1000)
		self._initialize_weights()

	def forward(self, x):
		x = self.model(x)
		x = x.view(-1, 1024)
		x = self.fc(x)
		return x
	#end forward

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
				elif isinstance(m, nn.BatchNorm2d):
					m.weight.data.fill_(1)
					m.bias.data.zero_()
				elif isinstance(m, nn.Linear):
					n = m.weight.size(1)
					m.weight.data.normal_(0, 0.01)
					m.bias.data.zero_()
            #end if
        #end for
    #end 

#end MobileNet

def mobilenet(**kwargs):
	model = MobileNet(False, False, **kwargs)
	return model
#end

def mobilenetr(**kwargs):
	model = MobileNet(True, False, **kwargs)
	return model
#end

def mobileneta(**kwargs):
	model = MobileNet(False, True, **kwargs)
	return model
#end

def mobilenetra(**kwargs):
	model = MobileNet(True, True, **kwargs)
	return model
#end

def mobilenetg4(**kwargs):
	model = MobileNet(False, False, 4, **kwargs)
	return model
#end

def mobilenetrg4(**kwargs):
	model = MobileNet(True, False, 4, **kwargs)
	return model
#end

def mobilenetag4(**kwargs):
	model = MobileNet(False, True, 4, **kwargs)
	return model
#end

def mobilenetrag4(**kwargs):
	model = MobileNet(True, True, 4, **kwargs)
	return model
#end
