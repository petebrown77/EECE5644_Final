?	B?Ѫ?D@B?Ѫ?D@!B?Ѫ?D@	`X?
??@`X?
??@!`X?
??@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'B?Ѫ?D@1?-?R\U??I(+?? @Y??ם?<??r0*	?(\??Au@2J
Iterator::Root::Map??
/???!.??	??Q@)B'?????1r?2V?O@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?M???
??!g?	?,@)-z?mà?1Jk??@#@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeath??W??!?n???#@)%#gaO;??1?/?_6 @:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2??2????!S?T?M?@)??2????1S?T?M?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;??Tގ??!??+?r@);??Tގ??1??+?r@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??<????!g??$?m<@)?_!se??1??i???@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?lW??e??!??4??N0@)??8?z?1?D?y???:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??À%w?!\?qT????)??À%w?1\?qT????:Preprocessing2E
Iterator::RootLnYk(??!??v??Q@)?1=a?t?1 ?SA??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?82.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9`X?
??@ICى?y?T@QΟ?h?^-@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	?-?R\U???-?R\U??!?-?R\U??*      ??!       2      ??!       :	(+?? @(+?? @!(+?? @B      ??!       J	??ם?<????ם?<??!??ם?<??R      ??!       Z	??ם?<????ם?<??!??ם?<??b      ??!       JGPUY`X?
??@b qCى?y?T@yΟ?h?^-@?"7
sequential/dense_1/MatMulMatMul?+@?|??!?+@?|??0"7
sequential/dense_2/MatMulMatMul:?J(????!h?? ????0"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum&?????!{?C????"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?_?????!<??vr??"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum?_?????!:???qk??"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum??3?DЃ?!
??%?_??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum˹??????!??[?ߤ??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentumW???????!????"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum???m??!=?z
ӂ??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum??>OZ??!S?????Q      Y@Y??}?	@@a"5?x+?P@qϴp?@y??O???"?

device?Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?82.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 