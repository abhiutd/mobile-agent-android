package com.example.android.mlmodelscopepredictor;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
//import snpe.Snpe;
import tflite.Tflite;
import tflite.PredictorData;
//import mpredictor.Mpredictor;
//import mpredictor.PredictorData;
import android.widget.Toast;

/** Classifies images */
public class ImageClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "MLModelScopePredictor";

  /** Name of the model file stored in Assets. */
  private static String FRAMEWORK = "XXX";
  private static String MODEL_PATH = "XXX";
  private static String HARDWARE = "XXX";
  private static String DATATYPE = "XXX";

  /** signal quantized model run*/
  private static Boolean QUANTIZED = false;

  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "labels.txt";
  public String LABEL_PATH_LOCAL;

  /** Data structures for profiling. */
  private boolean cold_start = true;
  private String MODEL_LOADING = "Model Loading: " + "...\n";
  private String DATA_PREPROCESSING = "Data Preprocessing: " + "...\n";
  private String MODEL_COLDSTART = "Model Coldstart: " + "...\n";
  private String MODEL_COMPUTATION = "Model Computation: " + "...\n";
  private String DATA_POSTPROCESSING = "Data Postprocessing: " + "...\n";
  private String INFERENCE_TIME = "Inference Latency: " + "...\n";
  private String THROUGHPUT = "Throughput: " + "...\n";

  private boolean record = true;
  private long model_loading = 0;
  private long model_coldstart = 0;
  private long[] data_preprocess = new long[100];
  private long[] model_compute = new long[100];
  private long[] data_postprocess = new long[100];
  private int num_of_inferences = 0;
  private long total_time = 0;

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  static final int DIM_IMG_SIZE_X = 224;
  static final int DIM_IMG_SIZE_Y = 224;

  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128.0f;


  /* Preallocated buffers for storing image data in. */
  private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

  /** Define framework predictor constructors */
  private tflite.PredictorData mytflitepredictor;

  private ByteBuffer imgData = null;

  ImageClassifier(Activity activity, String chosen_framework, String chosen_model, String chosen_hardware, String chosen_datatype) throws IOException {
    try{

      // Set framework, model and hardware choices
      FRAMEWORK = chosen_framework;
      MODEL_PATH = chosen_model;
      HARDWARE = chosen_hardware;
      DATATYPE = chosen_datatype;

      // DEBUG
      //Log.e(TAG, "In classifier object => Chosen framework: " + FRAMEWORK + ", Chosen model: " + MODEL_PATH + ", Chosen hardware: " + HARDWARE + ", Chosen datatype: " + DATATYPE);

      if(DATATYPE.equals("int8")) {
        QUANTIZED = true;
      } else {
        QUANTIZED = false;
      }

      // TRY temporary storage
      AssetManager assetManager = activity.getAssets();
      String abi = Build.CPU_ABI;
      String filesDir = activity.getFilesDir().getPath();
      String testPath = abi + "/" + MODEL_PATH;
      String testPathLabels = abi + "/" + LABEL_PATH;

      InputStream inStream = assetManager.open(MODEL_PATH);
      InputStream inStreamLabels = assetManager.open(LABEL_PATH);

      // Copy this file to an executable location
      File outFile = new File(filesDir, MODEL_PATH);
      File outFileLabels = new File(filesDir, LABEL_PATH);

      OutputStream outStream = new FileOutputStream(outFile);
      OutputStream outStreamLabels = new FileOutputStream(outFileLabels);

      byte[] buffer = new byte[1024];
      int read;
      while ((read = inStream.read(buffer)) != -1){
        outStream.write(buffer, 0, read);
      }
      byte[] bufferLabels = new byte[1024];
      int readLabels;
      while ((readLabels = inStreamLabels.read(bufferLabels)) != -1){
        outStreamLabels.write(bufferLabels, 0, readLabels);
      }

      inStream.close();
      outStream.flush();
      outStream.close();
      String tempPath = filesDir + "/" + MODEL_PATH;

      inStreamLabels.close();
      outStreamLabels.flush();
      outStreamLabels.close();
      String tempPathLabels = filesDir + "/" + LABEL_PATH;
      LABEL_PATH_LOCAL = tempPathLabels;

      long startTime_load = 0;
      long endTime_load = 0;
      if(FRAMEWORK.equals("Tensorflow Lite")) {
        // Create TfLite predictor
        startTime_load = SystemClock.uptimeMillis();
        switch (HARDWARE) {
          case "CPU_1_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_1_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 1 thread");
            break;
          case "CPU_2_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_2_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 2 threads");
            break;
          case "CPU_3_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_3_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 3 threads");
            break;
          case "CPU_4_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_4_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 4 threads");
            break;
          case "CPU_5_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_5_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 5 threads");
            break;
          case "CPU_6_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_6_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 6 threads");
            break;
          case "CPU_7_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_7_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 7 threads");
            break;
          case "CPU_8_thread":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_8_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: CPU 8 threads");
            break;
          case "GPU":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.GPU, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: GPU");
            break;
          case "NNAPI":
            mytflitepredictor = Tflite.new_(tempPath, Tflite.NNAPI, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: NNAPI");
            break;
          default:
            mytflitepredictor = Tflite.new_(tempPath, Tflite.CPU_1_thread, 1, false, false);
            Log.e(TAG, "In classifier object => created predictor with hw mode: default");
            break;
        }
        endTime_load = SystemClock.uptimeMillis();
        if(mytflitepredictor == null){
          Log.e(TAG, "Tflite.new_ returning null model");
        }
      }else if (FRAMEWORK.equals("Qualcomm SNPE")) {
        // Create SNPE predictor
        startTime_load = SystemClock.uptimeMillis();
        //mympredictor = Mpredictor.new_("Qualcomm SNPE", tempPath, Mpredictor.CPU_6_thread, 1, false, false);
        endTime_load = SystemClock.uptimeMillis();
        Log.e(TAG, "SNPE path broken!");
      }else {
        Log.e(TAG, "Invalid framework");
      }
      // INFO
      if(record){
        model_loading = endTime_load - startTime_load;
      }

    }catch (Exception e){
      e.printStackTrace();
    }
    imgData =
            ByteBuffer.allocateDirect(
                    4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
  }

  String classifyFrame(Bitmap bitmap) {

    if(FRAMEWORK.equals("Tensorflow Lite")) {
      if (mytflitepredictor == null) {
        Log.e(TAG, "Tensorflow lite Image classifier has not been initialized; Skipped.");
        return "Uninitialized Classifier.";
      }
    }else if(FRAMEWORK.equals("Qualcomm SNPE")) {
      //if (mysnpepredictor == null) {
      //  Log.e(TAG, "Qualcomm SNPE Image classifier has not been initialized; Skipped.");
      //  return "Uninitialized Classifier.";
      Log.e(TAG, "SNPE path broken!");
    }else {
      Log.e(TAG, "Invalid framework");
    }

    // DEBUG
    //Log.e(TAG, "In classifyFrame => Chosen framework: " + FRAMEWORK + ", Chosen model: " + MODEL_PATH + ", Chosen hardware: " + HARDWARE + ", Chosen datatype: " + DATATYPE + ", QUANTIZED: " + QUANTIZED);

    // read bitmapped frame into imgData (ByteBuffer)
    // and consequently into imgDataBytes (bytes[])
    long startTime_preprocessing = SystemClock.uptimeMillis();
    convertBitmapToByteBuffer(bitmap);
    // convert ByteBuffer[] into byte[]
    // as gomobile only supports []byte
    imgData.rewind();
    byte[] imgDataBytes = new byte[imgData.remaining()];
    try {
      imgData.get(imgDataBytes, 0, imgDataBytes.length);
    }catch (Exception e){
      e.printStackTrace();
    }
    long endTime_preprocessing = SystemClock.uptimeMillis();
    if(record){
      data_preprocess[num_of_inferences] = endTime_preprocessing - startTime_preprocessing;
    }

    // Separate cold start
    if(cold_start) {
      long startTime_coldstart = SystemClock.uptimeMillis();
      try {

        if(FRAMEWORK.equals("Tensorflow Lite")) {
          Tflite.predict(mytflitepredictor, imgDataBytes, QUANTIZED);
        }else if(FRAMEWORK.equals("Qualcomm SNPE")) {
          //Mpredictor.predict("Qualcomm SNPE", mympredictor, imgDataBytes, QUANTIZED);
          Log.e(TAG, "SNPE path broken!");
        }else {
          Log.e(TAG, "Invalid framework");
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
      long endTime_coldstart = SystemClock.uptimeMillis();
      if(record == true && cold_start == true){
        model_coldstart = endTime_coldstart - startTime_coldstart;
      }
    }
    cold_start = false;

    long startTime_computation = SystemClock.uptimeMillis();
    try {
      if(FRAMEWORK.equals("Tensorflow Lite")) {
        Tflite.predict(mytflitepredictor, imgDataBytes, QUANTIZED);
        //Mpredictor.predict("Tensorflow Lite", mympredictor, imgDataBytes, QUANTIZED);
      }else if(FRAMEWORK.equals("Qualcomm SNPE")) {
        //Mpredictor.predict("Qualcomm SNPE", mympredictor, imgDataBytes, QUANTIZED);
        Log.e(TAG, "SNPE path broken!");
      }else {
        Log.e(TAG, "Invalid framework");
      }
    }catch(Exception e){
      e.printStackTrace();
    }
    long endTime_computation = SystemClock.uptimeMillis();
    if(record){
      model_compute[num_of_inferences] = endTime_computation - startTime_computation;
    }

    String labelOutput = "";
    long startTime_postprocessing = SystemClock.uptimeMillis();
    try {
      if(FRAMEWORK.equals("Tensorflow Lite")) {
        labelOutput = Tflite.readPredictionOutput(mytflitepredictor, LABEL_PATH_LOCAL);
      }else if(FRAMEWORK.equals("Qualcomm SNPE")) {
        //labelOutput = Mpredictor.readPredictionOutput("Qualcomm SNPE", mympredictor, LABEL_PATH_LOCAL);
        Log.e(TAG, "SNPE path broken!");
      }else {
        Log.e(TAG, "Invalid framework");
      }
    }catch(Exception e){
      e.printStackTrace();
    }
    long endTime_postprocessing = SystemClock.uptimeMillis();
    if(record){
      data_postprocess[num_of_inferences] = endTime_postprocessing - startTime_postprocessing;
    }

    if(record) {

      total_time += data_preprocess[num_of_inferences] + model_compute[num_of_inferences] + data_postprocess[num_of_inferences];
      num_of_inferences++;
      Log.d(TAG, "Inference number: " + Long.toString((long)(num_of_inferences-1)));
      if (num_of_inferences == 100) {
        // compute model loading
        MODEL_LOADING = "Model Loading: " + Long.toString(model_loading) + " ms \n";
        Log.d(TAG, MODEL_LOADING);
        // compute model coldstart
        MODEL_COLDSTART = "Model Coldstart : " + Long.toString(model_coldstart) + " ms \n";
        Log.d(TAG, MODEL_COLDSTART);
        // compute data preprocessing, postprocessing and model compute
        long preprocess = 0;
        long compute = 0;
        long postprocess = 0;
        String PER_INFERENCE_LOG = "Inference";
        for (int i = 0; i < 100; i++) {
          preprocess += data_preprocess[i];
          compute += model_compute[i];
          postprocess += data_postprocess[i];
          PER_INFERENCE_LOG = "Inference number: " + Long.toString(i) + " => " + Long.toString(data_preprocess[i] + model_compute[i] + data_postprocess[i]) + " ms \n";
          Log.d(TAG, PER_INFERENCE_LOG);
        }
        DATA_PREPROCESSING = "Data Preprocessing : " + Long.toString(preprocess / 100) + " ms \n";
        Log.d(TAG, DATA_PREPROCESSING);
        MODEL_COMPUTATION = "Model Computation : " + Long.toString(compute / 100) + " ms \n";
        Log.d(TAG, MODEL_COMPUTATION);
        DATA_POSTPROCESSING = "Data Postprocessing: " + Long.toString(postprocess / 100) + " ms \n";
        Log.d(TAG, DATA_POSTPROCESSING);
        INFERENCE_TIME = "Inference Latency: " + Long.toString((preprocess + compute + postprocess)/100) + " ms \n";
        Log.d(TAG, INFERENCE_TIME);
        THROUGHPUT = "Throughput: " + Long.toString(1000 / ((preprocess/100) + (compute/100) + (postprocess/100))) + " inferences/sec \n";
        Log.d(TAG, THROUGHPUT);
        record = false;
      }
    }
    String textToShow = "Framework: " + FRAMEWORK + "\n" + "Model: " + MODEL_PATH + "\n" + "Hardware: " + HARDWARE + "\n" + "Datatype: " + DATATYPE + "\n" + "Top-5 predictions: " + labelOutput;
    textToShow = textToShow + "\n" + MODEL_LOADING + MODEL_COLDSTART + DATA_PREPROCESSING + MODEL_COMPUTATION + DATA_POSTPROCESSING + INFERENCE_TIME + THROUGHPUT;
    return textToShow;
  }

  public void close() {
    if(FRAMEWORK.equals("Tensorflow Lite")) {
      Tflite.close(mytflitepredictor);
      //Mpredictor.close("Tensorflow Lite", mympredictor);
    }else if(FRAMEWORK.equals("Qualcomm SNPE")) {
      Log.e(TAG, "SNPE path broken!");
    }else {
      Log.e(TAG, "Invalid framework");
    }
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();

    // check if passed bitmap is empty
    // DEBUG
    /*
    Bitmap emptyBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
    if(bitmap.sameAs(emptyBitmap)){
      // DEBUG
      Log.d(TAG, "Passed bitmap is also empty - what is happening ???");
    }*/
    
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    if(QUANTIZED == false) {
      // Convert the image to ByteBuffer floating point
      int pixel = 0;
      long startTime = SystemClock.uptimeMillis();
      for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
        for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
          final int val = intValues[pixel++];
          imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    } else {
      // Convert the image to Bytebuffer int
      int pixel = 0;
      long startTime = SystemClock.uptimeMillis();
      for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
        for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
          final int val = intValues[pixel++];
          imgData.putInt((val >> 16) & 0xFF);
          imgData.putInt((val >> 8) & 0xFF);
          imgData.putInt((val) & 0xFF);
        }
      }
    }
    long endTime = SystemClock.uptimeMillis();
  }

}
