package com.profaceswap

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var targetImage: ImageView
    private lateinit var selectTargetButton: Button
    private lateinit var selectSourceButton: Button
    private lateinit var swapButton: Button
    private lateinit var downloadButton: Button
    private lateinit var refreshButton: Button
    private lateinit var statusText: TextView

    private var targetBitmap: Bitmap? = null
    private var sourceBitmap: Bitmap? = null

    private var engine: FaceSwapEngine? = null

    private var hasSwapResult = false

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
        super.onCreate(
            savedInstanceState
        )

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

        downloadButton =
            findViewById(
                R.id.downloadButton
            )

        refreshButton =
            findViewById(
                R.id.refreshButton
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

        downloadButton.setOnClickListener {
            downloadResult()
        }

        refreshButton.setOnClickListener {
            refreshApp()
        }

        swapButton.isEnabled = false
        downloadButton.isEnabled = false
        selectTargetButton.isEnabled = false
        selectSourceButton.isEnabled = false

        statusText.text =
            "●  Loading AI models..."

        Thread {

            try {

                val newEngine =
                    FaceSwapEngine(
                        this@MainActivity
                    )

                if (
                    !newEngine.loadModels()
                ) {

                    newEngine.close()

                    runOnUiThread {

                        statusText.text =
                            "●  AI model loading failed"

                        selectTargetButton
                            .isEnabled = false

                        selectSourceButton
                            .isEnabled = false
                    }

                    return@Thread
                }

                engine =
                    newEngine

                runOnUiThread {

                    statusText.text =
                        "●  Ready to create"

                    selectTargetButton
                        .isEnabled = true

                    selectSourceButton
                        .isEnabled = true

                    updateSwapButton()
                }

            } catch (e: Throwable) {

                runOnUiThread {

                    statusText.text =
                        "●  Startup error: ${e.message}"
                }
            }

        }.start()
    }

    private fun loadTargetImage(
        uri: Uri
    ) {

        try {

            val bitmap =
                decodeBitmap(
                    uri
                )

            if (bitmap == null) {

                statusText.text =
                    "●  Could not open target image"

                return
            }

            targetBitmap?.recycle()

            targetBitmap =
                bitmap

            targetImage.setImageBitmap(
                bitmap
            )

            hasSwapResult =
                false

            downloadButton.isEnabled =
                false

            updateSwapButton()

            statusText.text =
                "●  Target photo selected"

        } catch (e: Throwable) {

            statusText.text =
                "●  Target image error: ${e.message}"
        }
    }

    private fun loadSourceImage(
        uri: Uri
    ) {

        try {

            val bitmap =
                decodeBitmap(
                    uri
                )

            if (bitmap == null) {

                statusText.text =
                    "●  Could not open source image"

                return
            }

            sourceBitmap?.recycle()

            sourceBitmap =
                bitmap

            hasSwapResult =
                false

            downloadButton.isEnabled =
                false

            updateSwapButton()

            statusText.text =
                "●  Source face selected"

        } catch (e: Throwable) {

            statusText.text =
                "●  Source image error: ${e.message}"
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
                "●  Select both images first"

            return
        }

        selectTargetButton.isEnabled =
            false

        selectSourceButton.isEnabled =
            false

        swapButton.isEnabled =
            false

        downloadButton.isEnabled =
            false

        refreshButton.isEnabled =
            false

        statusText.text =
            "●  Swapping faces..."

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

                    hasSwapResult =
                        true

                    downloadButton.isEnabled =
                        true

                    selectTargetButton.isEnabled =
                        true

                    selectSourceButton.isEnabled =
                        true

                    refreshButton.isEnabled =
                        true

                    statusText.text =
                        "●  Face swap complete"

                    updateSwapButton()
                }

            } catch (e: Throwable) {

                runOnUiThread {

                    statusText.text =
                        "●  Swap failed: ${e.message}"

                    selectTargetButton
                        .isEnabled = true

                    selectSourceButton
                        .isEnabled = true

                    refreshButton
                        .isEnabled = true

                    updateSwapButton()
                }
            }

        }.start()
    }

    private fun downloadResult() {

        val bitmap =
            targetBitmap

        if (
            bitmap == null ||
            !hasSwapResult
        ) {

            Toast.makeText(
                this,
                "Create a face swap first",
                Toast.LENGTH_SHORT
            ).show()

            return
        }

        downloadButton.isEnabled =
            false

        statusText.text =
            "●  Saving image..."

        Thread {

            try {

                val savedUri =
                    saveImageToGallery(
                        bitmap
                    )

                runOnUiThread {

                    downloadButton.isEnabled =
                        true

                    if (savedUri != null) {

                        statusText.text =
                            "●  Saved to Pictures/ProFaceSwap"

                        Toast.makeText(
                            this,
                            "Image saved to Gallery",
                            Toast.LENGTH_LONG
                        ).show()

                    } else {

                        statusText.text =
                            "●  Could not save image"

                        Toast.makeText(
                            this,
                            "Failed to save image",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

            } catch (e: Throwable) {

                runOnUiThread {

                    downloadButton.isEnabled =
                        true

                    statusText.text =
                        "●  Save failed: ${e.message}"

                    Toast.makeText(
                        this,
                        "Save failed",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

        }.start()
    }

    private fun saveImageToGallery(
        bitmap: Bitmap
    ): Uri? {

        val resolver =
            contentResolver

        val timestamp =
            SimpleDateFormat(
                "yyyyMMdd_HHmmss",
                Locale.US
            ).format(
                Date()
            )

        val fileName =
            "ProFaceSwap_$timestamp.jpg"

        val values =
            ContentValues().apply {

                put(
                    MediaStore.Images.Media.DISPLAY_NAME,
                    fileName
                )

                put(
                    MediaStore.Images.Media.MIME_TYPE,
                    "image/jpeg"
                )

                if (
                    Build.VERSION.SDK_INT >=
                    Build.VERSION_CODES.Q
                ) {

                    put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_PICTURES +
                                "/ProFaceSwap"
                    )

                    put(
                        MediaStore.Images.Media.IS_PENDING,
                        1
                    )
                }
            }

        val collection =
            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.Q
            ) {

                MediaStore.Images.Media.getContentUri(
                    MediaStore.VOLUME_EXTERNAL_PRIMARY
                )

            } else {

                MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            }

        val uri =
            resolver.insert(
                collection,
                values
            )
                ?: return null

        try {

            resolver.openOutputStream(
                uri
            ).use { output ->

                if (output == null) {

                    throw IllegalStateException(
                        "Could not open output stream"
                    )
                }

                if (
                    !bitmap.compress(
                        Bitmap.CompressFormat.JPEG,
                        95,
                        output
                    )
                ) {

                    throw IllegalStateException(
                        "Bitmap compression failed"
                    )
                }
            }

            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.Q
            ) {

                val completed =
                    ContentValues().apply {

                        put(
                            MediaStore.Images.Media.IS_PENDING,
                            0
                        )
                    }

                resolver.update(
                    uri,
                    completed,
                    null,
                    null
                )
            }

            return uri

        } catch (e: Throwable) {

            resolver.delete(
                uri,
                null,
                null
            )

            throw e
        }
    }

    private fun refreshApp() {

        targetBitmap?.recycle()
        sourceBitmap?.recycle()

        targetBitmap =
            null

        sourceBitmap =
            null

        hasSwapResult =
            false

        targetImage.setImageDrawable(
            null
        )

        downloadButton.isEnabled =
            false

        refreshButton.isEnabled =
            true

        statusText.text =
            "●  Ready for a new swap"

        updateSwapButton()
    }

    private fun decodeBitmap(
        uri: Uri
    ): Bitmap? {

        return contentResolver
            .openInputStream(
                uri
            )
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

        engine =
            null

        targetBitmap?.recycle()
        sourceBitmap?.recycle()

        targetBitmap =
            null

        sourceBitmap =
            null

        super.onDestroy()
    }
}
