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
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
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
  private static final int MAX_NUM_DETECTIONS = 10;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  // Config values.
  private static final int ROI_SIZE = 180;
  private Size roi_size;
  // Pre-allocated buffers.
  private final List<String> labels = new ArrayList<>();
  private Size inputSize;
  private Mat img_mat;
  private Mat img_roi;
  private org.opencv.imgproc.CLAHE clahe;

  private static final int CANNY_HIGH = 200;
  private static final int HOUGH_ACC_THRESHOLD = 85;
  private static final int HOUGH_MIN_DIST = 85;
  private static final int GAUSS_SIGMA = 3;

  // contains the scores of the detected coin
  float[][][][] inputArray;
  private float[][] outputScores;

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
   */
  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final Size inputSize)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    // load model
    MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelFilename);
    try (BufferedReader br =
                 new BufferedReader(
                         new InputStreamReader(context.getAssets().open(labelFilename)))) {
      String line;
      while ((line = br.readLine()) != null) {
        Log.w(TAG, line);
        d.labels.add(line);
      }
    }

    // set tflite options
    try {
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(NUM_THREADS);
      d.tfLite = new Interpreter(modelFile, options);
      d.tfLiteModel = modelFile;
      d.tfLiteOptions = options;
      d.inputSize = inputSize;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.img_mat = new Mat(d.inputSize, CvType.CV_8UC3);
    d.img_roi = new Mat(ROI_SIZE, ROI_SIZE, CvType.CV_8UC3);
    // clahe
    d.clahe = org.opencv.imgproc.Imgproc.createCLAHE(2, new Size(10, 10));
    d.inputArray = new float[1][ROI_SIZE][ROI_SIZE][3];
    d.outputScores = new float[1][d.labels.size()];
    d.roi_size = new Size(ROI_SIZE, ROI_SIZE);
    return d;
  }

  private int argmax(float[] array, int size) {
    double valmax = Double.MIN_VALUE;
    int argmax_idx = 0;
    for (int j = 0; j < size; ++j) {
      if (array[j] > valmax) {
        argmax_idx = j;
        valmax = array[j];
      }
    }

    return argmax_idx;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    Trace.beginSection("run");

    // bitmap to mat
    Utils.bitmapToMat(bitmap, img_mat);

    // rgb to hsv
    Mat img_hsv = new Mat(img_mat.size(), CvType.CV_8UC3);
    Imgproc.cvtColor(img_mat, img_hsv, Imgproc.COLOR_RGB2HSV);
    List<Mat> hsv_split = new ArrayList<>(3);
    org.opencv.core.Core.split(img_hsv, hsv_split);

    // apply histogram equalization on the `value` component of HSV
    clahe.apply(hsv_split.get(2), hsv_split.get(2));

    // merge the equalized image to rgb
    Mat img_equalized = new Mat(img_mat.size(), CvType.CV_8UC3);
    org.opencv.core.Core.merge(hsv_split, img_equalized);
    Imgproc.cvtColor(img_equalized, img_equalized, Imgproc.COLOR_HSV2RGB);

    // blur
    Mat img_gray = hsv_split.get(2);
    Mat img_gray_blurred = new Mat(img_mat.size(), CvType.CV_8UC1);
    org.opencv.imgproc.Imgproc.GaussianBlur(img_gray, img_gray_blurred, new Size(0, 0), GAUSS_SIGMA);

    // extract circles
    Mat circles = new Mat();
    Imgproc.HoughCircles(img_gray_blurred, circles, Imgproc.HOUGH_GRADIENT, 2, HOUGH_MIN_DIST, CANNY_HIGH, HOUGH_ACC_THRESHOLD);

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    float[][][][] inputArray = new float[1][ROI_SIZE][ROI_SIZE][3];
    float[][] outputArray = new float[1][labels.size()];

    for (int i = 0; (i < circles.cols()) && (i < MAX_NUM_DETECTIONS); ++i) {
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

      // check bounds
      if (detection.left < 0 || detection.right >= img_mat.width()
              || detection.top < 0 || detection.bottom >= img_mat.height())
        break;

      final Rect rect = new Rect((int) detection.left,(int) detection.top,
              2 * (int) radius,2 * (int) radius);

      // extract roi
      Mat img_roi_rect = new Mat(img_equalized, rect);

      // rescale ROI to 180 x 180 with bilinear interpolation
      Imgproc.resize(img_roi_rect, img_roi, roi_size);

      // copy to input array
      for (int row = 0; row < ROI_SIZE; row++) {
        for (int col = 0; col < ROI_SIZE; col++) {
          double[] rgb = img_roi.get(row, col);
          inputArray[0][row][col][0] = (float) rgb[0]; // R
          inputArray[0][row][col][1] = (float) rgb[1]; // G
          inputArray[0][row][col][2] = (float) rgb[2]; // B
        }
      }

      // predict
      tfLite.run(inputArray, outputArray);

      // find the best matching class
      float[] softmaxConfidences = outputArray[0];
      int argmax_idx = argmax(softmaxConfidences, labels.size());

      recognitions.add(new Recognition("" + i, labels.get(argmax_idx), softmaxConfidences[argmax_idx], detection));
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
