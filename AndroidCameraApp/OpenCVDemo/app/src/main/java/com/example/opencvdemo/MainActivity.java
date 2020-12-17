package com.example.opencvdemo;

import androidx.annotation.Nullable;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;

import android.graphics.Bitmap;

import android.os.Bundle;
import android.provider.MediaStore;

import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static int IMAGE_CAPTURE_CODE = 1;

    private ImageView mImageView;
    private Button mCaptureBtn;
    private Button mDetectBtn;

    private Bitmap mBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.imageView);
        mCaptureBtn = findViewById(R.id.captureBtn);
        mDetectBtn = findViewById(R.id.detectBtn);
        mDetectBtn.setEnabled(false);

        mCaptureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, IMAGE_CAPTURE_CODE);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode != IMAGE_CAPTURE_CODE)
            return;

        if (resultCode == RESULT_OK) {
            mBitmap = (Bitmap) data.getExtras().get("data");
            mImageView.setImageBitmap(mBitmap);
            mDetectBtn.setEnabled(true);
        } else if (resultCode == RESULT_CANCELED) {
            Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
        }
    }
}