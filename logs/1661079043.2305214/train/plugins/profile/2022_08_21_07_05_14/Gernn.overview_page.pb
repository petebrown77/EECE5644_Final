?	Ӿ?/@Ӿ?/@!Ӿ?/@	?z????	@?z????	@!?z????	@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ӿ?/@~?ƃ-v??1ګ??>??Iw稣?@Y }??A???r0*	;?O??`{@2J
Iterator::Root::Map??-Y??!??	?wN@)M??΢w??1?1???E@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2a??f??!????L1@)a??f??1????L1@:Preprocessing2E
Iterator::Root?c???_??!cx??+R@)?#?w~Q??1???>x'@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat)????u??!???:?&@)???j???1??A??M"@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF%u???!;?aGT?%@)|`?? ??1???AK?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice˻????!\????:@)˻????1\????:@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipʍ"k???!s??S;@)?P?,??1? ?s@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?H?H???!:??*?V*@)l
dv???1???'?@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorl
dv???!???'?@)l
dv???1???'?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?82.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?z????	@I??[???T@Q???"),@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~?ƃ-v??~?ƃ-v??!~?ƃ-v??      ??!       "	ګ??>??ګ??>??!ګ??>??*      ??!       2      ??!       :	w稣?@w稣?@!w稣?@B      ??!       J	 }??A??? }??A???! }??A???R      ??!       Z	 }??A??? }??A???! }??A???b      ??!       JGPUY?z????	@b q??[???T@y???"),@?"7
sequential/dense_1/MatMulMatMulȿ?8?̈́?!ȿ?8?̈́?0"7
sequential/dense_2/MatMulMatMulȿ?8?̈́?!ȿ?8?͔?0"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentumcy?|?C??!z|"w????"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul?,?5????!cI?7u??0"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentumd??cE???!???a?m??"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentumd??cE???!6???e??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum?ߒ????!wBۯ???"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum???^???!??ܞ???"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum???^???!*wbG???"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum???^???!?k&????Q      Y@Y??}?	@@a"5?x+?P@qv?XR@y????B???"?
device?Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?82.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?72.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 