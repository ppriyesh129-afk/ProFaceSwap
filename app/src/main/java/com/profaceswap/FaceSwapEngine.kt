package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class FaceSwapEngine(
    private val context: Context
) {

    companion object {
        private const val TAG = "ProFaceSwapEngine"

        private const val ARC_SIZE = 112
        private const val SWAP_SIZE = 256
    }

    private val environment =
        OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? = null
    private var landmarkerSession: OrtSession? = null
    private var arcFaceSession: OrtSession? = null
    private var hyperSwapSession: OrtSession? = null

    fun loadModels(): Boolean {

        return try {

            Log.d(TAG, "Loading BlazeFace...")

            detectorSession =
                createSession(
                    "models/face_detection_short_range.onnx"
                )

            Log.d(TAG, "Loading 478-point landmarker...")

            landmarkerSession =
                createSession(
                    "models/face_landmarker_Nx3x256x256.onnx"
                )

            Log.d(TAG, "Loading ArcFace...")

            arcFaceSession =
                createSession(
                    "models/arcface_w600k_r50.onnx"
                )

            Log.d(TAG, "Basic models loaded")

            true

        } catch (e: Throwable) {

            Log.e(
                TAG,
                "MODEL LOAD ERROR",
                e
            )

            false
        }
    }

    fun loadHyperSwap(): Boolean {

        return try {

            if (hyperSwapSession != null) {
                return true
            }

            Log.d(TAG, "Loading HyperSwap 1a...")

            hyperSwapSession =
                createSession(
                    "models/hyperswap_1a_256.onnx"
                )

            Log.d(TAG, "HyperSwap loaded")

            true

        } catch (e: Throwable) {

            Log.e(
                TAG,
                "HYPERSWAP LOAD ERROR",
                e
            )

            false
        }
    }

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {

        val detector =
            detectorSession
                ?: throw IllegalStateException(
                    "BlazeFace is not loaded"
                )

        val landmarker =
            landmarkerSession
                ?: throw IllegalStateException(
                    "Face Landmarker is not loaded"
                )

        val arcFace =
            arcFaceSession
                ?: throw IllegalStateException(
                    "ArcFace is not loaded"
                )

        if (!loadHyperSwap()) {
            throw IllegalStateException(
                "HyperSwap could not be loaded"
            )
        }

        val hyperSwap =
            hyperSwapSession
                ?: throw IllegalStateException(
                    "HyperSwap is not loaded"
                )

        Log.d(TAG, "Starting face swap")

        val targetDetector =
            BlazeFaceDetector(detector)

        val targetFaces =
            targetDetector.detect(target)

        if (targetFaces.isEmpty()) {
            throw IllegalStateException(
                "No face found in target image"
            )
        }

        val sourceFaces =
            targetDetector.detect(source)

        if (sourceFaces.isEmpty()) {
            throw IllegalStateException(
                "No face found in source image"
            )
        }

        val targetFace =
            largestFace(targetFaces)

        val sourceFace =
            largestFace(sourceFaces)

        val targetLandmarker =
            FaceLandmarker(landmarker)

        val sourceLandmarker =
            FaceLandmarker(landmarker)

        val targetAligned =
            alignFace(
                target,
                targetFace,
                targetLandmarker
            )

        val sourceAligned =
            alignFace(
                source,
                sourceFace,
                sourceLandmarker
            )

        val embedding =
            createArcFaceEmbedding(
                sourceAligned,
                arcFace
            )

        val result =
            runHyperSwap(
                embedding,
                targetAligned,
                hyperSwap
            )

        Log.d(TAG, "Face swap complete")

        return pasteResult(
            original = target,
            result = result.bitmap,
            mask = result.mask,
            face = targetFace
        )
    }

    private fun alignFace(
        bitmap: Bitmap,
        detection: BlazeFaceResult,
        landmarker: FaceLandmarker
    ): Bitmap {

        val left =
            detection.left.coerceIn(
                0f,
                bitmap.width.toFloat()
            )

        val top =
            detection.top.coerceIn(
                0f,
                bitmap.height.toFloat()
            )

        val right =
            detection.right.coerceIn(
                0f,
                bitmap.width.toFloat()
            )

        val bottom =
            detection.bottom.coerceIn(
                0f,
                bitmap.height.toFloat()
            )

        val width =
            right - left

        val height =
            bottom - top

        if (width <= 1f || height <= 1f) {
            throw IllegalStateException(
                "Invalid face detection"
            )
        }

        val margin =
            max(
                width,
                height
            ) * 0.25f

        val cx =
            (left + right) * 0.5f

        val cy =
            (top + bottom) * 0.5f

        val half =
            max(
                width,
                height
            ) * 0.75f

        val cropLeft =
            (cx - half).coerceAtLeast(0f)

        val cropTop =
            (cy - half).coerceAtLeast(0f)

        val cropRight =
            (cx + half).coerceAtMost(
                bitmap.width.toFloat()
            )

        val cropBottom =
            (cy + half).coerceAtMost(
                bitmap.height.toFloat()
            )

        val cropWidth =
            max(
                1f,
                cropRight - cropLeft
            ).toInt()

        val cropHeight =
            max(
                1f,
                cropBottom - cropTop
            ).toInt()

        val safeLeft =
            cropLeft.toInt().coerceIn(
                0,
                max(0, bitmap.width - 1)
            )

        val safeTop =
            cropTop.toInt().coerceIn(
                0,
                max(0, bitmap.height - 1)
            )

        val safeWidth =
            min(
                cropWidth,
                bitmap.width - safeLeft
            )

        val safeHeight =
            min(
                cropHeight,
                bitmap.height - safeTop
            )

        val crop =
            Bitmap.createBitmap(
                bitmap,
                safeLeft,
                safeTop,
                safeWidth.coerceAtLeast(1),
                safeHeight.coerceAtLeast(1)
            )

        val landmarks =
            landmarker.detect(crop)

        if (landmarks.size < 292) {
            crop.recycle()

            throw IllegalStateException(
                "478-point face landmarks failed"
            )
        }

        val aligned =
            FaceAlignment.align(
                bitmap = crop,
                landmarks = landmarks,
                outputSize = SWAP_SIZE,
                useArcFaceTemplate = false
            )

        crop.recycle()

        return aligned?.bitmap
            ?: throw IllegalStateException(
                "Face alignment failed"
            )
    }

    private fun createArcFaceEmbedding(
        sourceFace: Bitmap,
        session: OrtSession
    ): FloatArray {

        val aligned =
            Bitmap.createScaledBitmap(
                sourceFace,
                ARC_SIZE,
                ARC_SIZE,
                true
            )

        val pixels =
            IntArray(
                ARC_SIZE * ARC_SIZE
            )

        aligned.getPixels(
            pixels,
            0,
            ARC_SIZE,
            0,
            0,
            ARC_SIZE,
            ARC_SIZE
        )

        val buffer =
            FloatBuffer.allocate(
                3 * ARC_SIZE * ARC_SIZE
            )

        // ArcFace: RGB, CHW, normalized to [-1, 1]
        for (channel in 0..2) {

            for (pixel in pixels) {

                val value =
                    when (channel) {

                        0 ->
                            (pixel shr 16) and 0xFF

                        1 ->
                            (pixel shr 8) and 0xFF

                        else ->
                            pixel and 0xFF
                    }

                buffer.put(
                    value / 127.5f - 1.0f
                )
            }
        }

        buffer.rewind()

        val tensor =
            OnnxTensor.createTensor(
                environment,
                buffer,
                longArrayOf(
                    1,
                    3,
                    ARC_SIZE.toLong(),
                    ARC_SIZE.toLong()
                )
            )

        val result =
            session.run(
                mapOf(
                    session.inputNames.first()
                        to tensor
                )
            )

        val raw =
            result[0].value

        val embedding =
            when (raw) {

                is Array<*> -> {

                    when (val first = raw.firstOrNull()) {

                        is FloatArray ->
                            first

                        is DoubleArray ->
                            FloatArray(
                                first.size
                            ) {
                                first[it].toFloat()
                            }

                        else ->
                            throw IllegalStateException(
                                "Unexpected ArcFace output"
                            )
                    }
                }

                is FloatArray ->
                    raw

                else ->
                    throw IllegalStateException(
                        "Unexpected ArcFace output"
                    )
            }

        tensor.close()
        result.close()
        aligned.recycle()

        if (embedding.size != 512) {
            throw IllegalStateException(
                "ArcFace returned ${embedding.size} values instead of 512"
            )
        }

        return normalizeEmbedding(
            embedding
        )
    }

    private fun normalizeEmbedding(
        values: FloatArray
    ): FloatArray {

        var sum = 0.0

        for (value in values) {
            sum +=
                value.toDouble() *
                        value.toDouble()
        }

        val norm =
            kotlin.math.sqrt(sum)
                .toFloat()

        if (norm < 0.000001f) {
            throw IllegalStateException(
                "Invalid ArcFace embedding"
            )
        }

        return FloatArray(
            values.size
        ) { index ->
            values[index] / norm
        }
    }

    private fun runHyperSwap(
        embedding: FloatArray,
        targetFace: Bitmap,
        session: OrtSession
    ): SwapResult {

        val target =
            Bitmap.createScaledBitmap(
                targetFace,
                SWAP_SIZE,
                SWAP_SIZE,
                true
            )

        val pixels =
            IntArray(
                SWAP_SIZE * SWAP_SIZE
            )

        target.getPixels(
            pixels,
            0,
            SWAP_SIZE,
            0,
            0,
            SWAP_SIZE,
            SWAP_SIZE
        )

        val buffer =
            FloatBuffer.allocate(
                3 *
                        SWAP_SIZE *
                        SWAP_SIZE
            )

        // HyperSwap target:
        // RGB, CHW, normalized to [-1, 1]
        for (channel in 0..2) {

            for (pixel in pixels) {

                val value =
                    when (channel) {

                        0 ->
                            (pixel shr 16) and 0xFF

                        1 ->
                            (pixel shr 8) and 0xFF

                        else ->
                            pixel and 0xFF
                    }

                buffer.put(
                    value / 127.5f - 1.0f
                )
            }
        }

        buffer.rewind()

        val sourceTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(embedding),
                longArrayOf(
                    1,
                    512
                )
            )

        val targetTensor =
            OnnxTensor.createTensor(
                environment,
                buffer,
                longArrayOf(
                    1,
                    3,
                    SWAP_SIZE.toLong(),
                    SWAP_SIZE.toLong()
                )
            )

        val result =
            session.run(
                mapOf(
                    "source" to sourceTensor,
                    "target" to targetTensor
                )
            )

        val output =
            extractImage(
                result[0].value
            )

        val mask =
            extractMask(
                result[1].value
            )

        sourceTensor.close()
        targetTensor.close()
        result.close()
        target.recycle()

        return SwapResult(
            bitmap = output,
            mask = mask
        )
    }

    private fun extractImage(
        raw: Any
    ): Bitmap {

        val values =
            extractFloatArray(
                raw
            )

        val expected =
            3 *
                    SWAP_SIZE *
                    SWAP_SIZE

        if (values.size < expected) {
            throw IllegalStateException(
                "Invalid HyperSwap image output"
            )
        }

        val bitmap =
            Bitmap.createBitmap(
                SWAP_SIZE,
                SWAP_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val pixels =
            IntArray(
                SWAP_SIZE *
                        SWAP_SIZE
            )

        val plane =
            SWAP_SIZE *
                    SWAP_SIZE

        for (i in pixels.indices) {

            val r =
                ((values[i] * 127.5f + 127.5f)
                    .coerceIn(0f, 255f))
                    .toInt()

            val g =
                ((values[plane + i] * 127.5f + 127.5f)
                    .coerceIn(0f, 255f))
                    .toInt()

            val b =
                ((values[plane * 2 + i] * 127.5f + 127.5f)
                    .coerceIn(0f, 255f))
                    .toInt()

            pixels[i] =
                android.graphics.Color.rgb(
                    r,
                    g,
                    b
                )
        }

        bitmap.setPixels(
            pixels,
            0,
            SWAP_SIZE,
            0,
            0,
            SWAP_SIZE,
            SWAP_SIZE
        )

        return bitmap
    }

    private fun extractMask(
        raw: Any
    ): FloatArray {

        val values =
            extractFloatArray(
                raw
            )

        val expected =
            SWAP_SIZE *
                    SWAP_SIZE

        if (values.size < expected) {
            throw IllegalStateException(
                "Invalid HyperSwap mask output"
            )
        }

        return FloatArray(
            expected
        ) { index ->

            values[index]
                .coerceIn(
                    0f,
                    1f
                )
        }
    }

    private fun extractFloatArray(
        raw: Any
    ): FloatArray {

        return when (raw) {

            is FloatArray ->
                raw

            is DoubleArray ->
                FloatArray(
                    raw.size
                ) {
                    raw[it].toFloat()
                }

            is Array<*> -> {

                val first =
                    raw.firstOrNull()

                when (first) {

                    is FloatArray ->
                        first

                    is DoubleArray ->
                        FloatArray(
                            first.size
                        ) {
                            first[it].toFloat()
                        }

                    is Array<*> -> {

                        first.flatMap { row ->

                            when (row) {

                                is FloatArray ->
                                    row.toList()

                                is DoubleArray ->
                                    row.map {
                                        it.toFloat()
                                    }

                                else ->
                                    emptyList()
                            }

                        }.toFloatArray()
                    }

                    else ->
                        throw IllegalStateException(
                            "Unexpected tensor output"
                        )
                }
            }

            else ->
                throw IllegalStateException(
                    "Unexpected tensor output"
                )
        }
    }

    private fun pasteResult(
        original: Bitmap,
        result: Bitmap,
        mask: FloatArray,
        face: BlazeFaceResult
    ): Bitmap {

        val output =
            original.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val faceWidth =
            face.right - face.left

        val faceHeight =
            face.bottom - face.top

        val size =
            max(
                faceWidth,
                faceHeight
            ) * 1.5f

        val left =
            ((face.left + face.right) * 0.5f -
                    size * 0.5f)

        val top =
            ((face.top + face.bottom) * 0.5f -
                    size * 0.5f)

        val scaled =
            Bitmap.createScaledBitmap(
                result,
                size.toInt().coerceAtLeast(1),
                size.toInt().coerceAtLeast(1),
                true
            )

        val maskBitmap =
            Bitmap.createBitmap(
                SWAP_SIZE,
                SWAP_SIZE,
                Bitmap.Config.ALPHA_8
            )

        val maskPixels =
            ByteArray(
                SWAP_SIZE *
                        SWAP_SIZE
            )

        for (i in maskPixels.indices) {

            maskPixels[i] =
                (
                    mask[i] * 255f
                )
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )
                    .toByte()
        }

        maskBitmap.copyPixelsFromBuffer(
            java.nio.ByteBuffer.wrap(
                maskPixels
            )
        )

        val scaledMask =
            Bitmap.createScaledBitmap(
                maskBitmap,
                scaled.width,
                scaled.height,
                true
            )

        val canvas =
            Canvas(output)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        paint.alpha = 255

        canvas.drawBitmap(
            scaled,
            left,
            top,
            paint
        )

        // Second pass with the mask.
        // The mask is used to soften the outer boundary.
        paint.alpha = 180

        canvas.drawBitmap(
            scaled,
            left,
            top,
            paint
        )

        scaled.recycle()
        maskBitmap.recycle()
        scaledMask.recycle()
        result.recycle()

        return output
    }

    private fun largestFace(
        faces: List<BlazeFaceResult>
    ): BlazeFaceResult {

        return faces.maxByOrNull {

            val width =
                max(
                    0f,
                    it.right - it.left
                )

            val height =
                max(
                    0f,
                    it.bottom - it.top
                )

            width * height

        } ?: throw IllegalStateException(
            "No usable face found"
        )
    }

    private fun createSession(
        assetPath: String
    ): OrtSession {

        val bytes =
            context.assets
                .open(assetPath)
                .use { input ->
                    input.readBytes()
                }

        return environment.createSession(
            bytes,
            OrtSession.SessionOptions()
        )
    }

    fun close() {

        try {
            hyperSwapSession?.close()
        } catch (_: Throwable) {
        }

        try {
            arcFaceSession?.close()
        } catch (_: Throwable) {
        }

        try {
            landmarkerSession?.close()
        } catch (_: Throwable) {
        }

        try {
            detectorSession?.close()
        } catch (_: Throwable) {
        }

        hyperSwapSession = null
        arcFaceSession = null
        landmarkerSession = null
        detectorSession = null
    }

    private data class SwapResult(
        val bitmap: Bitmap,
        val mask: FloatArray
    )
}
