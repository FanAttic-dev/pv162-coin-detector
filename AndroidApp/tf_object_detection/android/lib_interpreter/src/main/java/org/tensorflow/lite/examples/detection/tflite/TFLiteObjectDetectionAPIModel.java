/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API: -
 * https://github.com/tensorflow/models/tree/master/research/object_detection where you can find the
 * training code.
 *
 * <p>To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithInterpreter";

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private static final int ROI_SIZE = 180;
  // Pre-allocated buffers.
  private final List<String> labels = new ArrayList<>();
  //private int[] intValues;
  // contains the scores of the detected coin
  private float[] outputScores;

  private ByteBuffer roi;

  private MappedByteBuffer tfLiteModel;
  private Interpreter.Options tfLiteOptions;
  private Interpreter tfLite;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
    /*
    MetadataExtractor metadata = new MetadataExtractor(modelFile);
    try (BufferedReader br =
        new BufferedReader(
            new InputStreamReader(
                metadata.getAssociatedFile(labelFilename), Charset.defaultCharset()))) {
      String line;
      while ((line = br.readLine()) != null) {
        Log.w(TAG, line);
        d.labels.add(line);
      }
    }
    */
    try (BufferedReader br =
                 new BufferedReader(
                         new InputStreamReader(context.getAssets().open(labelFilename)))) {
      String line;
      while ((line = br.readLine()) != null) {
        Log.w(TAG, line);
        d.labels.add(line);
      }
    }

    //d.inputSize = inputSize;

    try {
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(NUM_THREADS);
      d.tfLite = new Interpreter(modelFile, options);
      d.tfLiteModel = modelFile;
      d.tfLiteOptions = options;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.roi = ByteBuffer.allocateDirect(ROI_SIZE * ROI_SIZE * 3 * numBytesPerChannel);
    d.roi.order(ByteOrder.nativeOrder());
    //d.intValues = new int[d.inputSize * d.inputSize];

    d.outputScores = new float[d.labels.size()];
    return d;
  }

  private float[] softmax(float[] confidences) {
    float[] softmaxConfidences = new float[confidences.length];
    float total = 0;
    for (int i = 0; i < confidences.length; ++i) {
      total += Math.exp(confidences[i]);
    }
    for (int i = 0; i < confidences.length; ++i) {
      softmaxConfidences[i] = confidences[i] / total;
    }
    return softmaxConfidences;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    /*
    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap
    */

    Trace.beginSection("run");

    // TODO extract circles

    // bitmap to mat
    Mat mat = new Mat();
    Utils.bitmapToMat(bitmap, mat);
    // rgb to gray
    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
    org.opencv.imgproc.Imgproc.GaussianBlur(mat, mat, new Size(7,7), 1.5);

    Mat circles = new Mat();
    Imgproc.HoughCircles(mat, circles, Imgproc.HOUGH_GRADIENT, 2, 100, 400, 75);

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    //float[][] inputArray = new float[1][inputSize * inputSize * 3];
    float[][] outputArray = new float[1][labels.size()];

    int numDetections = circles.cols();

    for (int i = 0; i < numDetections; ++i) {
      double circle[] = circles.get(0, i);
      if (circle == null)
        break;

      float circleCenter[] = {(float) circle[0], (float) circle[1]};
      float radius = (float) circle[2];
      final RectF detection = new RectF(
              circleCenter[0] - radius,
              circleCenter[1] - radius,
              circleCenter[0] + radius,
              circleCenter[1] + radius);

      // TODO rescale values to [0, 1]
      // TODO rescale ROI to 180 x 180

      //tfLite.run(roi, outputArray);
      
      float[] softmaxConfidences = {0.2f, 0.4f, 0.8f, 0.1f, 0.3f, 0.1f};
      //softmaxConfidences = softmax(outputArray[0]);
      // TODO confidence
      float confidence = 0.f;
      int argmax = 0;
      for (int j = 0; j < labels.size(); ++j) {
        if (softmaxConfidences[j] > confidence) {
          argmax = j;
          confidence = softmaxConfidences[j];
        }
      }

      recognitions.add(new Recognition("" + i, labels.get(argmax), confidence, detection));
    }

    Trace.endSection(); // run

    Trace.endSection(); // recognizeImage
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (tfLite != null) {
      tfLite.close();
      tfLite = null;
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (tfLite != null) {
      tfLiteOptions.setNumThreads(numThreads);
      recreateInterpreter();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) {
      tfLiteOptions.setUseNNAPI(isChecked);
      recreateInterpreter();
    }
  }

  private void recreateInterpreter() {
    tfLite.close();
    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
  }
}
