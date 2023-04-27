import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_text as tf_text
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_data_validation as tfdv
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow_transform.beam import impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tfx_bsl.public import tfxio
import os
import uuid

class TFTVocab(object):

    TRAIN = "train"
    EVAL = "eval"

    def __init__(
        self,
        preprocessing_fn,
        fp_train,
        fp_eval,
        output_uri,
        params,
        schema_path=None,
        direct_runner=False,
    ):
        self.fp_train = fp_train
        self.fp_eval = fp_eval
        self.output_uri = output_uri
        self.preprocessing_fn = preprocessing_fn
        self.direct_runner = direct_runner
        self.schema_path = schema_path
        self.params = params

    def compute_stats(self):
        # direct runner
        stats = tfdv.generate_statistics_from_tfrecord(data_location=self.filepattern_train)
        return stats

    def maybe_compute_schema(self):
        if self.direct_runner:
            dirname = os.path.dirname(self.schema_uri)

        if self.schema_path is not None:
            schema = tfdv.load_schema_text(self.schema_path)
            tfdv.write_schema_text(schema, self.schema_uri)
            return schema

        if tf.io.gfile.exists(self.schema_uri):
            return tfdv.load_schema_text(self.schema_uri)

        stats = self.compute_stats()
        schema = tfdv.infer_schema(stats, infer_feature_shape=False)
        tfdv.write_schema_text(schema, self.schema_uri)
        return schema

    def run(self):

        schema = self.maybe_compute_schema()

        with beam.Pipeline(options=self.pipeline_options) as p:
            with impl.Context(self.temporary_dir):
                # Preprocess train data
                raw_train_dataset = self.tfr_reader(
                    p, self.filepattern_train, schema, self.TRAIN
                )
                transformed_train_dataset, transform_fn = self.analyze_and_transform(
                    raw_train_dataset, self.preprocessing_fn
                )
                self.write_tfrecords(
                    transformed_train_dataset,
                    self.transformed_train_uri,
                    self.TRAIN,
                )

                raw_eval_dataset = self.tfr_reader(
                    p, self.filepattern_eval, schema, self.EVAL
                )
                transformed_eval_dataset = self.transform(raw_eval_dataset, transform_fn)
                self.write_tfrecords(
                    transformed_eval_dataset, self.transformed_eval_uri, self.EVAL
                )

                self.write_transform_artifacts(transform_fn, self.transform_output_uri)

    def tfr_reader(self, pipeline, filepattern, schema, step):
        tfexample_tfxio = tfxio.TFExampleBeamRecord(physical_format=["tfrecord"], schema=schema)
        raw_data = pipeline | "ReadFromTFRecord {}".format(
            step
        ) >> beam.io.tfrecordio.ReadFromTFRecord(file_pattern=filepattern)
        parsed_raw_data = raw_data | "BeamSource {}".format(step) >> tfexample_tfxio.BeamSource()
        raw_metadata = tfexample_tfxio.TensorAdapterConfig()
        return (parsed_raw_data, raw_metadata)

    def analyze_and_transform(self, raw_dataset, preprocess_fn):
        (
            transformed_dataset,
            transform_fn,
        ) = raw_dataset | "AnalyzeAndTransformDataset" >> impl.AnalyzeAndTransformDataset(
            preprocess_fn
        )
        return transformed_dataset, transform_fn

    def transform(self, raw_dataset, transform_fn):
        transformed_dataset = (
            raw_dataset,
            transform_fn,
        ) | "TransformDataset" >> impl.TransformDataset()
        return transformed_dataset

    def write_tfrecords(self, transformed_dataset, output_dir, step):
        file_path_prefix = os.path.join(output_dir, "data")
        transformed_data, transformed_metadata = transformed_dataset
        (
            transformed_data
            | "Write {}".format(step)
            >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=file_path_prefix,
                file_name_suffix=".tfrecords",
                coder=example_proto_coder.ExampleProtoCoder(transformed_metadata.schema),
            )
        )

    def write_transform_artifacts(self, transform_fn, location):
        (transform_fn | "WriteTransformFn" >> transform_fn_io.WriteTransformFn(location))

    @property
    def temporary_dir(self):
        return os.path.join(self.output_uri, "temporary_dir")

    @property
    def staging_dir(self):
        return os.path.join(self.output_uri, "staging")

    @property
    def transformed_train_uri(self):
        return os.path.join(self.output_uri, "train")

    @property
    def transformed_eval_uri(self):
        return os.path.join(self.output_uri, "eval")

    @property
    def transform_output_uri(self):
        return os.path.join(self.output_uri, "transform_output")

    @property
    def tfdv_output_uri(self):
        return os.path.join(self.output_uri, "tfdv")

    @property
    def schema_uri(self):
        return os.path.join(self.transform_output_uri, "metadata", "schema.pbtxt")

    @property
    def pipeline_options(self):
        return PipelineOptions()

    def gen_job_name(self):
        job_name = "tft-transform-{}".format(str(uuid.uuid4()))
        print("job_name", job_name)
        return job_name