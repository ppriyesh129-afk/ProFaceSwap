package com.profaceswap

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ProFaceSwapStartup"
    }

    private lateinit var targetImage: ImageView
    private lateinit var statusText: TextView
    private lateinit var faceSwapEngine: FaceSwapEngine

    private var targetBitmap: Bitmap? = null
    private var sourceBitmap: Bitmap? = null

    private val targetPicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {

                targetBitmap = loadBitmap(uri)

                if (targetBitmap != null) {
                    targetImage.setImageBitmap(targetBitmap)
                    statusText.text = "Target image selected"
                } else {
                    statusText.text = "Could not load target image"
                }
            }
        }

    private val sourcePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {

                sourceBitmap = loadBitmap(uri)

                if (sourceBitmap != null) {
                    statusText.text = "Source face selected"
                } else {
                    statusText.text = "Could not load source image"
                }
            }
        }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(savedInstanceState)

        try {

            Log.d(
                TAG,
                "START: MainActivity.onCreate"
            )

            setContentView(
                R.layout.activity_main
            )

            Log.d(
                TAG,
                "OK: layout loaded"
            )

            targetImage =
                findViewById(
                    R.id.targetImage
                )

            statusText =
                findViewById(
                    R.id.statusText
                )

            Log.d(
                TAG,
                "OK: views loaded"
            )

            statusText.text =
                "Starting AI..."

            /*
             * IMPORTANT:
             * Do not load the 1.58 GB BlendSwap
             * model during app startup.
             *
             * We are testing whether the large
             * model initialization is causing the
             * immediate startup crash.
             */
            Thread {

                try {

                    Log.d(
                        TAG,
                        "START: FaceSwapEngine"
                    )

                    val engine =
                        FaceSwapEngine(
                            this@MainActivity
                        )

                    Log.d(
                        TAG,
                        "OK: FaceSwapEngine created"
                    )

                    val loaded =
                        engine.loadModels()

                    Log.d(
                        TAG,
                        "loadModels result = $loaded"
                    )

                    runOnUiThread {

                        faceSwapEngine =
                            engine

                        statusText.text =
                            if (loaded) {
                                "AI models loaded"
                            } else {
                                "AI model loading failed"
                            }
                    }

                } catch (e: Throwable) {

                    Log.e(
                        TAG,
                        "STARTUP AI ERROR",
                        e
                    )

                    runOnUiThread {

                        statusText.text =
                            "AI startup error: ${e.javaClass.simpleName}"
                    }
                }

            }.start()

            val selectTargetButton =
                findViewById<Button>(
                    R.id.selectTargetButton
                )

            val selectSourceButton =
                findViewById<Button>(
                    R.id.selectSourceButton
                )

            val swapButton =
                findViewById<Button>(
                    R.id.swapButton
                )

            selectTargetButton.setOnClickListener {

                targetPicker.launch(
                    "image/*"
                )
            }

            selectSourceButton.setOnClickListener {

                sourcePicker.launch(
                    "image/*"
                )
            }

            swapButton.setOnClickListener {

                if (!::faceSwapEngine.isInitialized) {

                    statusText.text =
                        "AI models are still loading"

                    return@setOnClickListener
                }

                val target =
                    targetBitmap

                val source =
                    sourceBitmap

                if (target == null) {

                    statusText.text =
                        "Please select a target image"

                    return@setOnClickListener
                }

                if (source == null) {

                    statusText.text =
                        "Please select a source face"

                    return@setOnClickListener
                }

                statusText.text =
                    "Detecting target face..."

                Thread {

                    try {

                        val result =
                            faceSwapEngine.processSwap(
                                target,
                                source
                            )

                        runOnUiThread {

                            targetImage.setImageBitmap(
                                result
                            )

                            statusText.text =
                                "478-point face analysis complete"
                        }

                    } catch (e: Throwable) {

                        Log.e(
                            TAG,
                            "SWAP ERROR",
                            e
                        )

                        runOnUiThread {

                            statusText.text =
                                "Processing error: ${e.javaClass.simpleName}"
                        }
                    }

                }.start()
            }

        } catch (e: Throwable) {

            Log.e(
                TAG,
                "MAIN ACTIVITY STARTUP ERROR",
                e
            )
        }
    }

    private fun loadBitmap(
        uri: Uri
    ): Bitmap? {

        return try {

            contentResolver
                .openInputStream(uri)
                ?.use { input ->

                    BitmapFactory.decodeStream(
                        input
                    )
                }

        } catch (e: Throwable) {

            Log.e(
                TAG,
                "IMAGE LOAD ERROR",
                e
            )

            null
        }
    }

    override fun onDestroy() {

        if (::faceSwapEngine.isInitialized) {
            faceSwapEngine.close()
        }

        targetBitmap?.recycle()
        sourceBitmap?.recycle()

        super.onDestroy()
    }
}
