?	??$??	@??$??	@!??$??	@	??~Mv	@??~Mv	@!??~Mv	@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??$??	@1?߆????I??[??@Y?????U??r0*	?Q??1|@2J
Iterator::Root::Map?>????!H????P@)8??w???1[Ҿ??rG@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?y?'L??!h??T
5@)?y?'L??1h??T
5@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat-ͭVc??!,ֻ?%@)n??KX??14)?;n? @:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateYL?Q??!????(@)6"????1????*h@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?m??ʆ??!N?q??@)?m??ʆ??1N?q??@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip	???W??!?I??	>@)	kc섗??1_k??C?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?????ױ?!Iⷔ?.@)???s?v??1k?ʃ	@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???+,??!Ⳛ??@)???+,??1Ⳛ??@:Preprocessing2E
Iterator::Root????2??!??J?}Q@)?(??/???1??A@9 @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?77.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??~Mv	@IU???VS@Q???Cw3@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	?߆?????߆????!?߆????*      ??!       2      ??!       :	??[??@??[??@!??[??@B      ??!       J	?????U???????U??!?????U??R      ??!       Z	?????U???????U??!?????U??b      ??!       JGPUY??~Mv	@b qU???VS@y???Cw3@?"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum}2?tk???!}2?tk???"7
sequential/dense_1/MatMulMatMulK??????!̾?
?)??0"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum?+?]???!M괜???"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum?c?̵???!<??????"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum?L??	???!cV??Z??"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentumK`XAW&??!;?q-x???"L
0gradient_tape/sequential/dense_1/MatMul/MatMul_1MatMul{?/h????!ʭwz4H??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum??m???!ߎX,?ܵ?"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul??m???!?o9ޏq??0"J
,gradient_tape/sequential/dense/MatMul/MatMulMatMul?\޵?d??!?;??.???0Q      Y@YK??">?@@a???`?P@q?????xS@y?F??Z}??"?
device?Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?77.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?77.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 