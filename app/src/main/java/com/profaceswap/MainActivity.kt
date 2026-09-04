package com.profaceswap

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        targetImage =
            findViewById(R.id.targetImage)

        statusText =
            findViewById(R.id.statusText)

        faceSwapEngine =
            FaceSwapEngine(this)

        statusText.text =
            if (faceSwapEngine.loadModels()) {
                "AI models loaded"
            } else {
                "Failed to load AI models"
            }

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
            targetPicker.launch("image/*")
        }

        selectSourceButton.setOnClickListener {
            sourcePicker.launch("image/*")
        }

        swapButton.setOnClickListener {

            val target = targetBitmap
            val source = sourceBitmap

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

                        targetImage.setImageBitmap(result)

                        statusText.text =
                            "478-point face analysis complete"
                    }

                } catch (e: Exception) {

                    e.printStackTrace()

                    runOnUiThread {
                        statusText.text =
                            "Processing error: ${e.message}"
                    }
                }

            }.start()
        }
    }

    private fun loadBitmap(uri: Uri): Bitmap? {

        return try {

            contentResolver
                .openInputStream(uri)
                ?.use { input ->
                    BitmapFactory.decodeStream(input)
                }

        } catch (e: Exception) {

            e.printStackTrace()
            null
        }
    }

    override fun onDestroy() {

        faceSwapEngine.close()

        targetBitmap?.recycle()
        sourceBitmap?.recycle()

        super.onDestroy()
    }
}
