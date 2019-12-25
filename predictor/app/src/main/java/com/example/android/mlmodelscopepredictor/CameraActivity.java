package com.example.android.mlmodelscopepredictor;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.system.Os;

/** Main {@code Activity} class for the Camera app. */
public class CameraActivity extends Activity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera);
    Intent intent = getIntent();
    String framework = intent.getStringExtra("framework");
    String model = intent.getStringExtra("model");
    String hardware = intent.getStringExtra("hardware");
    String datatype = intent.getStringExtra("datatype");
    try {
      // set environment variable for qualcomm snpe symphony library
      Os.setenv("SYMPHONY_INIT_TEST", "SYMPHONY_YES", true);
    }catch(Exception e){
      e.printStackTrace();
    }
    // DEBUG
    System.out.println("In CameraActivity -- User wants to run "+model+" deployed in "+framework+" on mobile "+hardware+" with datatype "+datatype);
    if (null == savedInstanceState) {
      getFragmentManager()
          .beginTransaction()
          .replace(R.id.container, Camera2BasicFragment.newInstance(framework, model, hardware, datatype))
          .commit();
    }
  }
}
