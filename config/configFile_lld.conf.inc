[componentInstances:cComponentManager]
instance[frames].type = cFramer
instance[pitchComponent].type = cPitchDirection

[frames:cFramer]
reader.dmLevel = wave
writer.dmLevel = frames
;\{\cm[bufferModeRbConf{shared/BufferModeRb.conf.inc}:path to included config to set the buffer mode for the standard ringbuffer levels]}
copyInputName = 1
frameMode = fixed
frameSize = 1.0
frameStep = 0.0
frameCenterSpecial = left
noPostEOIprocessing = 1

[pitchComponent:cPitchDirection]
reader.dmLevel = frames
writer.dmLevel = pitchComponent
\{\cm[bufferModeRbLagConf{shared/BufferModeRbLag.conf.inc}]}
copyInputName = 1
buffersize_sec= 1.0
directionScore = 0