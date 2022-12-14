?	??$???@??$???@!??$???@	h??n??@h??n??@!h??n??@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??$???@1[rPB??I????@Y? {???r0*	??ʡE?{@2J
Iterator::Root::Map??]?p??!#qEU?N@)??(]???1?x|???F@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2;%?α?!???;?m/@);%?α?1???;?m/@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ???-???!?|M???1@)?,_?????1????)@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat6Φ#????!???? *@)t]?@???1 ???$%@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?F?????!m???B?@)?F?????1m???B?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?͋_???!??? ?wB@)ꗈ?ο??1???O?@
@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?!S>U??!?	?9?4@)-σ??v??1e??}<@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor'i????!??2?+?@)'i????1??2?+?@:Preprocessing2E
Iterator::Root????????!x`?2?O@)>?$@M-{?1﫧2????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9g??n??@I?$:??S@Q??DZ2@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	[rPB??[rPB??![rPB??*      ??!       2      ??!       :	????@????@!????@B      ??!       J	? {???? {???!? {???R      ??!       Z	? {???? {???!? {???b      ??!       JGPUYg??n??@b q?$:??S@y??DZ2@?"9
sequential_1/dense_6/MatMulMatMul[??Q???![??Q???0"9
sequential_1/dense_7/MatMulMatMul?å?ބ?!?mv???0"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum??(????!?i?U???"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum1?Qރ?!??
?m??"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum??p,???!cZ?&
V??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum??p,???!???B?>??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum??1?u???!KSY???"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum??1?u???!?H?H??"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentumY?ݩ?{??!?ٿr??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentumY?ݩ?{??!X?5?7???Q      Y@Y??}?	@@a"5?x+?P@q?mr`U@y??????"?
device?Your program is NOT input-bound because only 3.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?78.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?85.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 