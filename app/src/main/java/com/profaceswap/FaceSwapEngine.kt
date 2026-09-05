package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.sqrt

class FaceSwapEngine(
    private val context: Context
) {

    companion object {
        private const val TAG = "ProFaceSwapEngine"

        private const val ARC_SIZE = 112
        private const val HYPER_SIZE = 256
    }

    private val environment =
        OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? = null
    private var landmarkerSession: OrtSession? = null
    private var arcFaceSession: OrtSession? = null
    private var hyperSwapSession: OrtSession? = null

    fun loadModels(): Boolean {

        return try {

            android.util.Log.d(
                TAG,
                "Loading BlazeFace"
            )

            detectorSession =
                createSession(
                    "models/face_detection_short_range.onnx"
                )

            android.util.Log.d(
                TAG,
                "Loading 478-point landmarker"
            )

            landmarkerSession =
                createSession(
                    "models/face_landmarker_Nx3x256x256.onnx"
                )

            android.util.Log.d(
                TAG,
                "Loading ArcFace"
            )

            arcFaceSession =
                createSession(
                    "models/arcface_w600k_r50.onnx"
                )

            android.util.Log.d(
                TAG,
                "Basic models loaded"
            )

            true

        } catch (e: Throwable) {

            android.util.Log.e(
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

            android.util.Log.d(
                TAG,
                "Loading HyperSwap 1a 256"
            )

            hyperSwapSession =
                createSession(
                    "models/hyperswap_1a_256.onnx"
                )

            android.util.Log.d(
                TAG,
                "HyperSwap loaded"
            )

            true

        } catch (e: Throwable) {

            android.util.Log.e(
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

        android.util.Log.d(
            TAG,
            "Detecting source face"
        )

        val faceDetector =
            BlazeFaceDetector(
                detector
            )

        val sourceFaces =
            faceDetector.detect(
                source
            )

        if (sourceFaces.isEmpty()) {

            throw IllegalStateException(
                "No face found in source image"
            )
        }

        android.util.Log.d(
            TAG,
            "Detecting target face"
        )

        val targetFaces =
            faceDetector.detect(
                target
            )

        if (targetFaces.isEmpty()) {

            throw IllegalStateException(
                "No face found in target image"
            )
        }

        val sourceFace =
            largestFace(
                sourceFaces
            )

        val targetFace =
            largestFace(
                targetFaces
            )

        android.util.Log.d(
            TAG,
            "Preparing source face"
        )

        val sourceAligned =
            prepareFace(
                source,
                sourceFace,
                landmarker
            )

        android.util.Log.d(
            TAG,
            "Preparing target face"
        )

        val targetAligned =
            prepareFace(
                target,
                targetFace,
                landmarker
            )

        try {

            android.util.Log.d(
                TAG,
                "Creating ArcFace embedding"
            )

            val sourceEmbedding =
                createArcFaceEmbedding(
                    sourceAligned.bitmap,
                    arcFace
                )

            android.util.Log.d(
                TAG,
                "Running HyperSwap"
            )

            val swap =
                runHyperSwap(
                    sourceEmbedding,
                    targetAligned.bitmap,
                    hyperSwap
                )

            android.util.Log.d(
                TAG,
                "Pasting HyperSwap result"
            )

            return pasteBack(
                original = target,
                swap = swap,
                transform = targetAligned.transform,
                targetLandmarks = targetAligned.landmarks
            )

        } finally {

            sourceAligned.bitmap.recycle()
            targetAligned.bitmap.recycle()
        }
    }

    private fun prepareFace(
        bitmap: Bitmap,
        detection: BlazeFaceResult,
        landmarkerSession: OrtSession
    ): PreparedFace {

        val faceLandmarker =
            FaceLandmarker(
                landmarkerSession
            )

        val width =
            detection.right -
                    detection.left

        val height =
            detection.bottom -
                    detection.top

        val faceSize =
            max(
                width,
                height
            )

        if (faceSize <= 1f) {

            throw IllegalStateException(
                "Invalid face size"
            )
        }

        val roiSize =
            faceSize * 1.5f

        val centerX =
            (
                detection.left +
                        detection.right
                ) * 0.5f

        val centerY =
            (
                detection.top +
                        detection.bottom
                ) * 0.5f

        var roiLeft =
            centerX -
                    roiSize * 0.5f

        var roiTop =
            centerY -
                    roiSize * 0.5f

        var roiRight =
            centerX +
                    roiSize * 0.5f

        var roiBottom =
            centerY +
                    roiSize * 0.5f

        if (roiLeft < 0f) {

            val difference =
                -roiLeft

            roiLeft = 0f
            roiRight += difference
        }

        if (roiTop < 0f) {

            val difference =
                -roiTop

            roiTop = 0f
            roiBottom += difference
        }

        if (
            roiRight >
            bitmap.width.toFloat()
        ) {

            val difference =
                roiRight -
                        bitmap.width.toFloat()

            roiRight =
                bitmap.width.toFloat()

            roiLeft -= difference
        }

        if (
            roiBottom >
            bitmap.height.toFloat()
        ) {

            val difference =
                roiBottom -
                        bitmap.height.toFloat()

            roiBottom =
                bitmap.height.toFloat()

            roiTop -= difference
        }

        roiLeft =
            roiLeft.coerceIn(
                0f,
                bitmap.width.toFloat() - 1f
            )

        roiTop =
            roiTop.coerceIn(
                0f,
                bitmap.height.toFloat() - 1f
            )

        roiRight =
            roiRight.coerceIn(
                roiLeft + 1f,
                bitmap.width.toFloat()
            )

        roiBottom =
            roiBottom.coerceIn(
                roiTop + 1f,
                bitmap.height.toFloat()
            )

        val cropLeft =
            roiLeft.toInt()

        val cropTop =
            roiTop.toInt()

        val cropWidth =
            max(
                1,
                (
                    roiRight -
                            roiLeft
                    ).toInt()
            ).coerceAtMost(
                bitmap.width -
                        cropLeft
            )

        val cropHeight =
            max(
                1,
                (
                    roiBottom -
                            roiTop
                    ).toInt()
            ).coerceAtMost(
                bitmap.height -
                        cropTop
            )

        val crop =
            Bitmap.createBitmap(
                bitmap,
                cropLeft,
                cropTop,
                cropWidth,
                cropHeight
            )

        val rawLandmarks =
            faceLandmarker.detect(
                crop
            )

        if (
            rawLandmarks.size <
            292
        ) {

            crop.recycle()

            throw IllegalStateException(
                "Could not obtain 478 face landmarks"
            )
        }

        val landmarks =
            Array(
                rawLandmarks.size
            ) { index ->

                floatArrayOf(

                    rawLandmarks[index][0] *
                            crop.width /
                            256f,

                    rawLandmarks[index][1] *
                            crop.height /
                            256f,

                    if (
                        rawLandmarks[index].size >
                        2
                    ) {

                        rawLandmarks[index][2]

                    } else {

                        0f
                    }
                )
            }

        for (point in landmarks) {

            point[0] +=
                cropLeft.toFloat()

            point[1] +=
                cropTop.toFloat()
        }

        crop.recycle()

        val matrix =
            calculateSimilarityTransform(
                FaceAlignment.fivePoints(
                    landmarks
                ),
                hyperSwapTemplate()
            )

        val aligned =
            warpBitmap(
                bitmap = bitmap,
                matrix = matrix,
                outputSize = HYPER_SIZE
            )

        val inverse =
            Matrix()

        if (!matrix.invert(inverse)) {

            aligned.recycle()

            throw IllegalStateException(
                "Could not invert alignment matrix"
            )
        }

        return PreparedFace(
            bitmap = aligned,
            transform = matrix,
            landmarks = landmarks
        )
    }

    private fun hyperSwapTemplate():
        Array<FloatArray> {

        return arrayOf(

            floatArrayOf(
                84.87f,
                105.94f
            ),

            floatArrayOf(
                171.13f,
                105.94f
            ),

            floatArrayOf(
                128.00f,
                146.66f
            ),

            floatArrayOf(
                96.95f,
                188.64f
            ),

            floatArrayOf(
                159.05f,
                188.64f
            )
        )
    }

    private fun warpBitmap(
        bitmap: Bitmap,
        matrix: Matrix,
        outputSize: Int
    ): Bitmap {

        val output =
            Bitmap.createBitmap(
                outputSize,
                outputSize,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(
                output
            )

        canvas.drawColor(
            Color.BLACK
        )

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            bitmap,
            matrix,
            paint
        )

        return output
    }

    private fun calculateSimilarityTransform(
        source: Array<FloatArray>,
        target: Array<FloatArray>
    ): Matrix {

        if (
            source.size != 5 ||
            target.size != 5
        ) {

            throw IllegalStateException(
                "Five landmarks are required"
            )
        }

        var sourceCx = 0.0
        var sourceCy = 0.0

        var targetCx = 0.0
        var targetCy = 0.0

        for (i in 0 until 5) {

            sourceCx +=
                source[i][0].toDouble()

            sourceCy +=
                source[i][1].toDouble()

            targetCx +=
                target[i][0].toDouble()

            targetCy +=
                target[i][1].toDouble()
        }

        sourceCx /= 5.0
        sourceCy /= 5.0

        targetCx /= 5.0
        targetCy /= 5.0

        var a = 0.0
        var b = 0.0
        var denominator = 0.0

        for (i in 0 until 5) {

            val sx =
                source[i][0].toDouble() -
                        sourceCx

            val sy =
                source[i][1].toDouble() -
                        sourceCy

            val tx =
                target[i][0].toDouble() -
                        targetCx

            val ty =
                target[i][1].toDouble() -
                        targetCy

            a +=
                sx * tx +
                        sy * ty

            b +=
                sx * ty -
                        sy * tx

            denominator +=
                sx * sx +
                        sy * sy
        }

        if (
            denominator <
            0.000001
        ) {

            throw IllegalStateException(
                "Face alignment failed"
            )
        }

        val magnitude =
            sqrt(
                a * a +
                        b * b
            )

        if (
            magnitude <
            0.000001
        ) {

            throw IllegalStateException(
                "Invalid alignment magnitude"
            )
        }

        val scale =
            magnitude /
                    denominator

        val cos =
            a /
                    magnitude

        val sin =
            b /
                    magnitude

        val translateX =
            targetCx -
                    scale *
                    (
                        cos *
                                sourceCx -
                                sin *
                                sourceCy
                        )

        val translateY =
            targetCy -
                    scale *
                    (
                        sin *
                                sourceCx +
                                cos *
                                sourceCy
                        )

        val matrix =
            Matrix()

        matrix.setValues(
            floatArrayOf(

                (scale * cos)
                    .toFloat(),

                (-scale * sin)
                    .toFloat(),

                translateX
                    .toFloat(),

                (scale * sin)
                    .toFloat(),

                (scale * cos)
                    .toFloat(),

                translateY
                    .toFloat(),

                0f,
                0f,
                1f
            )
        )

        return matrix
    }

    private fun createArcFaceEmbedding(
        face: Bitmap,
        session: OrtSession
    ): FloatArray {

        val resized =
            Bitmap.createScaledBitmap(
                face,
                ARC_SIZE,
                ARC_SIZE,
                true
            )

        val pixels =
            IntArray(
                ARC_SIZE *
                        ARC_SIZE
            )

        resized.getPixels(
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
                3 *
                        ARC_SIZE *
                        ARC_SIZE
            )

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
                    value /
                            127.5f -
                            1f
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

        val inputName =
            session.inputNames.first()

        val result =
            session.run(
                mapOf(
                    inputName to
                            tensor
                )
            )

        val embedding =
            extractFloatArray(
                result[0].value
            )

        tensor.close()
        result.close()
        resized.recycle()

        if (
            embedding.size !=
            512
        ) {

            throw IllegalStateException(
                "ArcFace returned " +
                        "${embedding.size} values"
            )
        }

        return normalizeEmbedding(
            embedding
        )
    }

    private fun runHyperSwap(
        embedding: FloatArray,
        targetFace: Bitmap,
        session: OrtSession
    ): HyperSwapResult {

        val pixels =
            IntArray(
                HYPER_SIZE *
                        HYPER_SIZE
            )

        targetFace.getPixels(
            pixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        val buffer =
            FloatBuffer.allocate(
                3 *
                        HYPER_SIZE *
                        HYPER_SIZE
            )

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
                    value /
                            127.5f -
                            1f
                )
            }
        }

        buffer.rewind()

        val sourceTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(
                    embedding
                ),
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
                    HYPER_SIZE.toLong(),
                    HYPER_SIZE.toLong()
                )
            )

        val inputNames =
            session.inputNames.toList()

        android.util.Log.d(
            TAG,
            "HyperSwap inputs: $inputNames"
        )

        val inputMap =
            HashMap<String, OnnxTensor>()

        if (
            inputNames.contains(
                "source"
            )
        ) {

            inputMap["source"] =
                sourceTensor

        } else {

            inputMap[
                inputNames[0]
            ] =
                sourceTensor
        }

        if (
            inputNames.contains(
                "target"
            )
        ) {

            inputMap["target"] =
                targetTensor

        } else {

            if (inputNames.size < 2) {

                sourceTensor.close()
                targetTensor.close()

                throw IllegalStateException(
                    "HyperSwap expects fewer than " +
                            "2 inputs: $inputNames"
                )
            }

            inputMap[
                inputNames[1]
            ] =
                targetTensor
        }

        val result =
            try {

                session.run(
                    inputMap
                )

            } catch (e: Throwable) {

                sourceTensor.close()
                targetTensor.close()

                throw IllegalStateException(
                    "HyperSwap inference failed: " +
                            "${e.message}",
                    e
                )
            }

        try {

            val outputInfo =
                session.outputInfo

            val diagnostics =
                StringBuilder()

            diagnostics.append(
                "HyperSwap output count: "
            )

            diagnostics.append(
                result.size()
            )

            diagnostics.append(
                "\n\n"
            )

            diagnostics.append(
                "MODEL OUTPUT INFO:\n"
            )

            for (
                entry in outputInfo.entries
            ) {

                diagnostics.append(
                    entry.key
                )

                diagnostics.append(
                    " -> "
                )

                diagnostics.append(
                    entry.value.info
                )

                diagnostics.append(
                    "\n"
                )
            }

            diagnostics.append(
                "\nACTUAL OUTPUTS:\n"
            )

            for (
                index in 0 until result.size()
            ) {

                val value =
                    result[index].value

                diagnostics.append(
                    "Output "
                )

                diagnostics.append(
                    index
                )

                diagnostics.append(
                    ": "
                )

                diagnostics.append(
                    value?.javaClass?.name
                        ?: "null"
                )

                diagnostics.append(
                    "\n"
                )

                if (
                    value is FloatArray
                ) {

                    diagnostics.append(
                        "  FloatArray size="
                    )

                    diagnostics.append(
                        value.size
                    )

                    diagnostics.append(
                        "\n"
                    )

                } else if (
                    value is DoubleArray
                ) {

                    diagnostics.append(
                        "  DoubleArray size="
                    )

                    diagnostics.append(
                        value.size
                    )

                    diagnostics.append(
                        "\n"
                    )

                } else if (
                    value is Array<*>
                ) {

                    diagnostics.append(
                        "  Array size="
                    )

                    diagnostics.append(
                        value.size
                    )

                    diagnostics.append(
                        "\n"
                    )

                    val first =
                        value.firstOrNull()

                    if (
                        first != null
                    ) {

                        diagnostics.append(
                            "  First element type="
                        )

                        diagnostics.append(
                            first.javaClass.name
                        )

                        diagnostics.append(
                            "\n"
                        )

                        when (first) {

                            is FloatArray -> {

                                diagnostics.append(
                                    "  First FloatArray size="
                                )

                                diagnostics.append(
                                    first.size
                                )

                                diagnostics.append(
                                    "\n"
                                )
                            }

                            is DoubleArray -> {

                                diagnostics.append(
                                    "  First DoubleArray size="
                                )

                                diagnostics.append(
                                    first.size
                                )

                                diagnostics.append(
                                    "\n"
                                )
                            }

                            is Array<*> -> {

                                diagnostics.append(
                                    "  First nested Array size="
                                )

                                diagnostics.append(
                                    first.size
                                )

                                diagnostics.append(
                                    "\n"
                                )
                            }
                        }
                    }
                }

                diagnostics.append(
                    "\n"
                )
            }

            android.util.Log.d(
                TAG,
                diagnostics.toString()
            )

            val imageRaw =
                if (
                    result.size() >
                    0
                ) {

                    result[0].value

                } else {

                    null
                }

            val maskRaw =
                if (
                    result.size() >
                    1
                ) {

                    result[1].value

                } else {

                    null
                }

            if (
                imageRaw == null
            ) {

                throw IllegalStateException(
                    diagnostics.toString()
                )
            }

            val image =
                try {

                    extractFloatArray(
                        imageRaw
                    )

                } catch (e: Throwable) {

                    throw IllegalStateException(
                        diagnostics.toString() +
                                "\nIMAGE DECODE ERROR: " +
                                "${e.message}",
                        e
                    )
                }

            val expectedImageSize =
                3 *
                        HYPER_SIZE *
                        HYPER_SIZE

            if (
                image.size !=
                expectedImageSize
            ) {

                throw IllegalStateException(
                    diagnostics.toString() +
                            "\nEXPECTED IMAGE VALUES: " +
                            expectedImageSize +
                            "\nACTUAL IMAGE VALUES: " +
                            image.size
                )
            }

            if (
                maskRaw == null
            ) {

                throw IllegalStateException(
                    diagnostics.toString() +
                            "\nHyperSwap did not return " +
                            "a second mask output."
                )
            }

            val mask =
                try {

                    extractFloatArray(
                        maskRaw
                    )

                } catch (e: Throwable) {

                    throw IllegalStateException(
                        diagnostics.toString() +
                                "\nMASK DECODE ERROR: " +
                                "${e.message}",
                        e
                    )
                }

            val expectedMaskSize =
                HYPER_SIZE *
                        HYPER_SIZE

            if (
                mask.size <
                expectedMaskSize
            ) {

                throw IllegalStateException(
                    diagnostics.toString() +
                            "\nEXPECTED MASK VALUES: " +
                            expectedMaskSize +
                            "\nACTUAL MASK VALUES: " +
                            mask.size
                )
            }

            return HyperSwapResult(
                bitmap =
                    imageToBitmap(
                        image
                    ),
                mask = mask
            )

        } finally {

            sourceTensor.close()
            targetTensor.close()
            result.close()
        }
    }

    private fun pasteBack(
        original: Bitmap,
        swap: HyperSwapResult,
        transform: Matrix,
        targetLandmarks: Array<FloatArray>
    ): Bitmap {

        val output =
            original.copy(Bitmap.Config.ARGB_8888, true)

        val inverse = Matrix()

        if (!transform.invert(inverse)) {
            swap.bitmap.recycle()
            throw IllegalStateException("Could not invert paste transform")
        }

        val colorMatched =
            matchSkinTone(
                swap.bitmap,
                original,
                targetLandmarks
            )

        val maskedSwap =
            createMaskedSwap(
                colorMatched,
                swap.mask,
                targetLandmarks,
                transform
            )

        colorMatched.recycle()

        val paint = Paint(
            Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
        )

        paint.isDither = true
        paint.alpha = 255

        Canvas(output).drawBitmap(maskedSwap, inverse, paint)

        smoothFaceEdges(output, targetLandmarks)

        maskedSwap.recycle()
        swap.bitmap.recycle()

        return output
    }

    private fun smoothFaceEdges(
        bitmap: Bitmap,
        landmarks: Array<FloatArray>
    ) {

        val canvas = Canvas(bitmap)

        val path = Path()

        val jaw = listOf(
            10,338,297,332,284,251,389,356,
            454,323,361,288,397,365,379,
            378,400,377,152,148,176,149,
            150,136,172,58,132,93,234,
            127,162,21,54,103,67
        )

        if (jaw.isEmpty()) return

        path.moveTo(
            landmarks[jaw.first()][0],
            landmarks[jaw.first()][1]
        )

        for (i in 1 until jaw.size) {
            path.lineTo(
                landmarks[jaw[i]][0],
                landmarks[jaw[i]][1]
            )
        }

        path.close()

        val blurPaint = Paint(Paint.ANTI_ALIAS_FLAG)

        blurPaint.style = Paint.Style.STROKE
        blurPaint.strokeWidth = 14f
        blurPaint.maskFilter =
            android.graphics.BlurMaskFilter(
                10f,
                android.graphics.BlurMaskFilter.Blur.NORMAL
            )

        blurPaint.color = Color.TRANSPARENT

        canvas.drawPath(path, blurPaint)
    }

    private fun matchSkinTone(
        swap: Bitmap,
        target: Bitmap,
        landmarks: Array<FloatArray>
    ): Bitmap {

        val result =
            swap.copy(Bitmap.Config.ARGB_8888, true)

        var sr = 0L
        var sg = 0L
        var sb = 0L

        var tr = 0L
        var tg = 0L
        var tb = 0L

        var count = 0L

        val step = 6

        for (y in 0 until HYPER_SIZE step step) {
            for (x in 0 until HYPER_SIZE step step) {

                val sc = swap.getPixel(x, y)

                sr += Color.red(sc)
                sg += Color.green(sc)
                sb += Color.blue(sc)

                val tx =
                    (landmarks[1][0] + (x - 128)).toInt()
                        .coerceIn(0, target.width - 1)

                val ty =
                    (landmarks[1][1] + (y - 128)).toInt()
                        .coerceIn(0, target.height - 1)

                val tc = target.getPixel(tx, ty)

                tr += Color.red(tc)
                tg += Color.green(tc)
                tb += Color.blue(tc)

                count++
            }
        }

        if (count == 0L) return result

        val rScale = tr.toFloat() / sr.coerceAtLeast(1)
        val gScale = tg.toFloat() / sg.coerceAtLeast(1)
        val bScale = tb.toFloat() / sb.coerceAtLeast(1)

        val pixels =
            IntArray(HYPER_SIZE * HYPER_SIZE)

        result.getPixels(
            pixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        for (i in pixels.indices) {

            val c = pixels[i]

            val r =
                (Color.red(c) * rScale)
                    .toInt()
                    .coerceIn(0, 255)

            val g =
                (Color.green(c) * gScale)
                    .toInt()
                    .coerceIn(0, 255)

            val b =
                (Color.blue(c) * bScale)
                    .toInt()
                    .coerceIn(0, 255)

            pixels[i] =
                Color.argb(
                    Color.alpha(c),
                    r,
                    g,
                    b
                )
        }

        result.setPixels(
            pixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        return result
    }

    private fun createMaskedSwap(
        bitmap: Bitmap,
        mask: FloatArray,
        targetLandmarks: Array<FloatArray>,
        transform: Matrix
    ): Bitmap {

        val expected =
            HYPER_SIZE *
                    HYPER_SIZE

        if (
            mask.size <
            expected
        ) {

            throw IllegalStateException(
                "Invalid HyperSwap mask: " +
                        "${mask.size} values"
            )
        }

        val sourcePixels =
            IntArray(
                expected
            )

        bitmap.getPixels(
            sourcePixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        val targetMaskBitmap =
            createTargetFaceMask(
                targetLandmarks = targetLandmarks,
                transform = transform,
                width = HYPER_SIZE,
                height = HYPER_SIZE
            )

        val targetMaskPixels =
            ByteArray(
                expected
            )

        targetMaskBitmap.copyPixelsToBuffer(
            java.nio.ByteBuffer.wrap(
                targetMaskPixels
            )
        )

        targetMaskBitmap.recycle()

        val outputPixels =
            IntArray(
                expected
            )

        for (
            i in 0 until expected
        ) {

            val sourceColor =
                sourcePixels[i]

            val hyperAlpha =
                mask[i]
                    .coerceIn(
                        0f,
                        1f
                    )

            val landmarkAlpha =
                (
                    targetMaskPixels[i]
                        .toInt() and 0xFF
                    ) / 255f

            /*
             * Keep the HyperSwap mask as the main mask,
             * but allow the target face shape to extend it.
             *
             * This helps when the source face is thinner
             * than the target face.
             */

            val landmarkContribution =
                landmarkAlpha * 0.65f

            val combinedAlpha =
                (
                    hyperAlpha +
                            landmarkContribution *
                            (1f - hyperAlpha)
                    )
                    .coerceIn(
                        0f,
                        1f
                    )

            val alpha =
                (
                    combinedAlpha *
                            255f
                    )
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )

            outputPixels[i] =
                Color.argb(
                    alpha,
                    Color.red(
                        sourceColor
                    ),
                    Color.green(
                        sourceColor
                    ),
                    Color.blue(
                        sourceColor
                    )
                )
        }

        val output =
            Bitmap.createBitmap(
                HYPER_SIZE,
                HYPER_SIZE,
                Bitmap.Config.ARGB_8888
            )

        output.setPixels(
            outputPixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        return output
    }

    private fun imageToBitmap(
        values: FloatArray
    ): Bitmap {

        val plane =
            HYPER_SIZE *
                    HYPER_SIZE

        val expected =
            plane * 3

        if (
            values.size !=
            expected
        ) {

            throw IllegalStateException(
                "Invalid HyperSwap image output: " +
                        "expected $expected values, " +
                        "got ${values.size}"
            )
        }

        val pixels =
            IntArray(
                plane
            )

        for (
            i in 0 until plane
        ) {

            val r =
                (
                    values[i] *
                            127.5f +
                            127.5f
                    )
                    .coerceIn(
                        0f,
                        255f
                    )
                    .toInt()

            val g =
                (
                    values[plane + i] *
                            127.5f +
                            127.5f
                    )
                    .coerceIn(
                        0f,
                        255f
                    )
                    .toInt()

            val b =
                (
                    values[plane * 2 + i] *
                            127.5f +
                            127.5f
                    )
                    .coerceIn(
                        0f,
                        255f
                    )
                    .toInt()

            pixels[i] =
                Color.rgb(
                    r,
                    g,
                    b
                )
        }

        val bitmap =
            Bitmap.createBitmap(
                HYPER_SIZE,
                HYPER_SIZE,
                Bitmap.Config.ARGB_8888
            )

        bitmap.setPixels(
            pixels,
            0,
            HYPER_SIZE,
            0,
            0,
            HYPER_SIZE,
            HYPER_SIZE
        )

        return bitmap
    }

    private fun extractFloatArray(raw: Any): FloatArray {

        fun flatten(value: Any?): MutableList<Float> {
            val out = mutableListOf<Float>()

            when (value) {
                is FloatArray -> out.addAll(value.toList())
                is DoubleArray -> out.addAll(value.map { it.toFloat() })
                is Array<*> -> value.forEach { out.addAll(flatten(it)) }
                null -> {}
                else -> throw IllegalStateException(
                    "Unexpected tensor type: ${value.javaClass.name}"
                )
            }

            return out
        }

        return flatten(raw).toFloatArray()
    }

    private fun normalizeEmbedding(
        values: FloatArray
    ): FloatArray {

        var sum = 0.0

        for (
            value in values
        ) {

            sum +=
                value.toDouble() *
                        value.toDouble()
        }

        val norm =
            sqrt(
                sum
            ).toFloat()

        if (
            norm <
            0.000001f
        ) {

            throw IllegalStateException(
                "Invalid ArcFace embedding"
            )
        }

        return FloatArray(
            values.size
        ) { index ->

            values[index] /
                    norm
        }
    }

    private fun largestFace(
        faces: List<BlazeFaceResult>
    ): BlazeFaceResult {

        return faces.maxByOrNull {

            val width =
                max(
                    0f,
                    it.right -
                            it.left
                )

            val height =
                max(
                    0f,
                    it.bottom -
                            it.top
                )

            width *
                    height

        } ?: throw IllegalStateException(
            "No usable face found"
        )
    }

    private fun createSession(
        assetPath: String
    ): OrtSession {

        val file =
            java.io.File(
                context.cacheDir,
                assetPath.substringAfterLast(
                    "/"
                )
            )

        if (
            !file.exists()
        ) {

            context.assets
                .open(
                    assetPath
                )
                .use { input ->

                    file.outputStream()
                        .use { output ->

                            input.copyTo(
                                output
                            )
                        }
                }
        }

        return environment.createSession(
            file.absolutePath,
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

    private fun convexHullPoints(
        points: FloatArray
    ): List<PointF> {

        val input =
            ArrayList<PointF>()

        var i = 0

        while (i + 1 < points.size) {

            input.add(
                PointF(
                    points[i],
                    points[i + 1]
                )
            )

            i += 2
        }

        if (input.size <= 2) {
            return input
        }

        val sorted =
            input.sortedWith(
                compareBy<PointF> {
                    it.x
                }.thenBy {
                    it.y
                }
            )

        fun cross(
            a: PointF,
            b: PointF,
            c: PointF
        ): Float {

            return (
                (b.x - a.x) *
                        (c.y - a.y) -
                        (b.y - a.y) *
                        (c.x - a.x)
                )
        }

        val lower =
            ArrayList<PointF>()

        for (point in sorted) {

            while (
                lower.size >= 2 &&
                cross(
                    lower[lower.size - 2],
                    lower[lower.size - 1],
                    point
                ) <= 0f
            ) {

                lower.removeAt(
                    lower.size - 1
                )
            }

            lower.add(point)
        }

        val upper =
            ArrayList<PointF>()

        for (index in sorted.indices.reversed()) {

            val point =
                sorted[index]

            while (
                upper.size >= 2 &&
                cross(
                    upper[upper.size - 2],
                    upper[upper.size - 1],
                    point
                ) <= 0f
            ) {

                upper.removeAt(
                    upper.size - 1
                )
            }

            upper.add(point)
        }

        lower.removeAt(
            lower.size - 1
        )

        upper.removeAt(
            upper.size - 1
        )

        lower.addAll(
            upper
        )

        return lower
    }

    private fun createTargetFaceMask(
        targetLandmarks: Array<FloatArray>,
        transform: Matrix,
        width: Int,
        height: Int
    ): Bitmap {

        val mask =
            Bitmap.createBitmap(
                width,
                height,
                Bitmap.Config.ALPHA_8
            )

        val canvas =
            Canvas(mask)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )

        paint.color = Color.WHITE
        paint.style = Paint.Style.FILL

        val points =
            FloatArray(
                targetLandmarks.size * 2
            )

        for (i in targetLandmarks.indices) {

            points[i * 2] =
                targetLandmarks[i][0]

            points[i * 2 + 1] =
                targetLandmarks[i][1]
        }

        transform.mapPoints(points)

        val hull =
            convexHullPoints(points)

        if (hull.size >= 3) {

            val centerX =
                hull.map { it.x }.average().toFloat()

            val centerY =
                hull.map { it.y }.average().toFloat()

            val shrink =
                0.96f

            for (point in hull) {

                point.x =
                    centerX +
                            (point.x - centerX) *
                            shrink

                point.y =
                    centerY +
                            (point.y - centerY) *
                            shrink
            }

            val path =
                Path()

            path.moveTo(
                hull[0].x,
                hull[0].y
            )

            for (i in 1 until hull.size) {

                path.lineTo(
                    hull[i].x,
                    hull[i].y
                )
            }

            path.close()

            canvas.drawPath(
                path,
                paint
            )

            val featherPaint =
                Paint(
                    Paint.ANTI_ALIAS_FLAG
                )

            featherPaint.color = Color.WHITE
            featherPaint.style = Paint.Style.FILL
            featherPaint.maskFilter =
                android.graphics.BlurMaskFilter(
                    12f,
                    android.graphics.BlurMaskFilter.Blur.NORMAL
                )

            canvas.drawPath(
                path,
                featherPaint
            )
        }

        return mask
    }

    private data class PreparedFace(
        val bitmap: Bitmap,
        val transform: Matrix,
        val landmarks: Array<FloatArray>
    )

    private data class HyperSwapResult(
        val bitmap: Bitmap,
        val mask: FloatArray
    )
}
