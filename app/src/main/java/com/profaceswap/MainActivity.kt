package com.profaceswap

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var targetImage: ImageView
    private lateinit var selectTargetButton: Button
    private lateinit var selectSourceButton: Button
    private lateinit var swapButton: Button
    private lateinit var statusText: TextView

    private var targetBitmap: Bitmap? = null
    private var sourceBitmap: Bitmap? = null

    private var engine: FaceSwapEngine? = null

    private val targetPicker =
        registerForActivityResult(
            ActivityResultContracts.PickVisualMedia()
        ) { uri ->

            if (uri != null) {
                loadTargetImage(uri)
            }
        }

    private val sourcePicker =
        registerForActivityResult(
            ActivityResultContracts.PickVisualMedia()
        ) { uri ->

            if (uri != null) {
                loadSourceImage(uri)
            }
        }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        targetImage =
            findViewById(
                R.id.targetImage
            )

        selectTargetButton =
            findViewById(
                R.id.selectTargetButton
            )

        selectSourceButton =
            findViewById(
                R.id.selectSourceButton
            )

        swapButton =
            findViewById(
                R.id.swapButton
            )

        statusText =
            findViewById(
                R.id.statusText
            )

        selectTargetButton.setOnClickListener {

            targetPicker.launch(
                PickVisualMediaRequest(
                    ActivityResultContracts
                        .PickVisualMedia
                        .ImageOnly
                )
            )
        }

        selectSourceButton.setOnClickListener {

            sourcePicker.launch(
                PickVisualMediaRequest(
                    ActivityResultContracts
                        .PickVisualMedia
                        .ImageOnly
                )
            )
        }

        swapButton.setOnClickListener {

            startFaceSwap()
        }

        swapButton.isEnabled = false

        statusText.text =
            "Loading AI models..."

        Thread {

            try {

                val newEngine =
                    FaceSwapEngine(
                        this@MainActivity
                    )

                if (!newEngine.loadModels()) {

                    newEngine.close()

                    runOnUiThread {

                        statusText.text =
                            "AI model loading failed"

                        selectTargetButton.isEnabled =
                            false

                        selectSourceButton.isEnabled =
                            false
                    }

                    return@Thread
                }

                engine =
                    newEngine

                runOnUiThread {

                    statusText.text =
                        "Ready — select target and source"

                    selectTargetButton.isEnabled =
                        true

                    selectSourceButton.isEnabled =
                        true
                }

            } catch (e: Throwable) {

                runOnUiThread {

                    statusText.text =
                        "Startup error: ${e.message}"
                }
            }

        }.start()
    }

    private fun loadTargetImage(
        uri: Uri
    ) {

        try {

            val bitmap =
                decodeBitmap(uri)

            if (bitmap == null) {

                statusText.text =
                    "Could not open target image"

                return
            }

            targetBitmap?.recycle()

            targetBitmap =
                bitmap

            targetImage.setImageBitmap(
                bitmap
            )

            updateSwapButton()

            statusText.text =
                "Target image selected"

        } catch (e: Throwable) {

            statusText.text =
                "Target image error: ${e.message}"
        }
    }

    private fun loadSourceImage(
        uri: Uri
    ) {

        try {

            val bitmap =
                decodeBitmap(uri)

            if (bitmap == null) {

                statusText.text =
                    "Could not open source image"

                return
            }

            sourceBitmap?.recycle()

            sourceBitmap =
                bitmap

            updateSwapButton()

            statusText.text =
                "Source face selected"

        } catch (e: Throwable) {

            statusText.text =
                "Source image error: ${e.message}"
        }
    }

    private fun updateSwapButton() {

        swapButton.isEnabled =
            targetBitmap != null &&
                    sourceBitmap != null &&
                    engine != null
    }

    private fun startFaceSwap() {

        val target =
            targetBitmap

        val source =
            sourceBitmap

        val currentEngine =
            engine

        if (
            target == null ||
            source == null ||
            currentEngine == null
        ) {

            statusText.text =
                "Select both images first"

            return
        }

        selectTargetButton.isEnabled =
            false

        selectSourceButton.isEnabled =
            false

        swapButton.isEnabled =
            false

        statusText.text =
            "Swapping faces..."

        Thread {

            try {

                val result =
                    currentEngine.processSwap(
                        target = target,
                        source = source
                    )

                runOnUiThread {

                    targetImage.setImageBitmap(
                        result
                    )

                    targetBitmap?.recycle()

                    targetBitmap =
                        result

                    statusText.text =
                        "Face swap complete"

                    selectTargetButton.isEnabled =
                        true

                    selectSourceButton.isEnabled =
                        true

                    updateSwapButton()
                }

            } catch (e: Throwable) {

                runOnUiThread {

                    statusText.text =
                        "Swap failed: ${e.message}"

                    selectTargetButton.isEnabled =
                        true

                    selectSourceButton.isEnabled =
                        true

                    updateSwapButton()
                }
            }
        }.start()
    }

    private fun decodeBitmap(
        uri: Uri
    ): Bitmap? {

        return contentResolver
            .openInputStream(uri)
            ?.use { input ->

                BitmapFactory.decodeStream(
                    input
                )
            }
    }

    override fun onDestroy() {

        try {
            engine?.close()
        } catch (_: Throwable) {
        }

        engine = null

        targetBitmap?.recycle()
        sourceBitmap?.recycle()

        targetBitmap = null
        sourceBitmap = null

        super.onDestroy()
    }
}
