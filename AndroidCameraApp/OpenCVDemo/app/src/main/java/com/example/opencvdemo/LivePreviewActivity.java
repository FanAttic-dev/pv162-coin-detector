package com.example.opencvdemo;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.ActivityManager;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.FrameLayout;

public class LivePreviewActivity extends AppCompatActivity {

    private static final String[] PERMISSIONS = {
            Manifest.permission.CAMERA
    };

    private static final int REQUEST_PERMISSIONS = 34;

    private static int IMAGE_CAPTURE_CODE = 1;

    private boolean isCameraInitialized = false;

    private static Camera mCamera;

    private static CameraPreview mCameraPreview;

    private FrameLayout mFrameLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onResume() {
        super.onResume();

        if (arePermissionsDenied()) {
            requestPermissions(PERMISSIONS, REQUEST_PERMISSIONS);
        }

        if (!isCameraInitialized) {
            initCamera();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    private boolean arePermissionsDenied() {
        for (String permission : PERMISSIONS) {
            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED)
                return true;
        }
        return false;
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_PERMISSIONS && grantResults.length > 0) {
            if (arePermissionsDenied()) {
                ((ActivityManager) (this.getSystemService(ACTIVITY_SERVICE))).clearApplicationUserData();
                recreate();
            } else {
                onResume();
            }
        }
    }

    private void initCamera() {
        if (isCameraInitialized)
            return;

        mCamera = Camera.open();
        mCameraPreview = new CameraPreview(this, mCamera);
        mFrameLayout = findViewById(R.id.frameView);
        mFrameLayout.addView(mCameraPreview);
        rotateCamera();

        Camera.Parameters params = mCamera.getParameters();
        params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        mCamera.setParameters(params);
    }

    private void rotateCamera() {
        if (mCamera == null)
            return;

        int rotation = this.getWindowManager().getDefaultDisplay().getRotation();
        switch (rotation) {
            case 0:
                rotation = 90;
                break;
            case 1:
                rotation = 0;
                break;
            case 2:
                rotation = 270;
                break;
            default:
                rotation = 180;
                break;
        }

        mCamera.setDisplayOrientation(rotation);
    }

    @Override
    protected void onPause() {
        super.onPause();

        releaseCamera();
    }

    private void releaseCamera() {
        if (mCamera == null)
            return;

        mFrameLayout.removeView(mCameraPreview);
        mCamera.release();
        mCamera = null;
    }
}