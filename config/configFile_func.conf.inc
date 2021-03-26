[componentInstances:cComponentManager]
instance[percentilesFunctional].type=cFunctionals

[percentilesFunctional:cFunctionals]
reader.dmLevel = frames
writer.dmLevel = percentilesFunctional
\{\cm[bufferModeRbConf{shared/BufferModeRb.conf.inc}:]}
copyInputName = 1
\{\cm[frameModeFunctionalsConf{shared/FrameModeFunctionals.conf.inc}:]}
functionalsEnabled = Percentiles
Percentiles.percentile[0] = 0.26
Percentiles.pctlrange[0] = 0-1
Percentiles.interp = 0*7