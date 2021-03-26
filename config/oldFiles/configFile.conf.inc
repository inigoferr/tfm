[componentInstances:cComponentManager]
//instance[percentilesFunctional].type=cFunctionals
instance[frame].type = cFramer
instance[loudness].type = cIntensity
instance[pitchComponent].type = cPitchDirection


[frame:cFramer]
reader.dmLevel = wave
writer.dmLevel = frame
\{\cm[bufferModeRbConf{shared/BufferModeRb.conf.inc}:path to included config to set the buffer mode for the standard ringbuffer levels]}
frameMode = fixed
frameSize = 1
frameCenterSpecial = left
\{\cm[frameModeFunctionalsConf{shared/FrameModeFunctionals.conf.inc}:path to included config to set frame mode for all functionals]}

[loudness:cIntensity]
reader.dmLevel = frame
writer.dmLevel = loudness
\{\cm[bufferModeRbConf]}


[pitchComponent:cPitchDirection]
reader.dmLevel = frame
writer.dmLevel = pitchComponent
\{\cm[bufferModeRbLagConf{shared/BufferModeRbLag.conf.inc}]}
copyInputName = 1
buffersize_sec= 1.0
directionScore = 0

