[componentInstances:cComponentManager]
instance[percentilesFunctional].type=cFunctionals

[percentilesFunctional:cFunctionals]
reader.dmLevel = wave
writer.dmLevel = percentilesFunctional
\{\cm[bufferModeRbConf{BufferMode.conf.inc}:]}
copyInputName = 1
\{\cm[frameModeFunctionalsConf{FrameModeFunctionals.conf.inc}:]}
functionalsEnabled = Percentiles
Percentiles.quartiles = 0
Percentiles.iqr = 0
Percentiles.percentile[0] = 0.20
Percentiles.percentile[1] = 0.50
Percentiles.percentile[2] = 0.80
Percentiles.pctlrange[0] = 0-2
Percentiles.interp = 1