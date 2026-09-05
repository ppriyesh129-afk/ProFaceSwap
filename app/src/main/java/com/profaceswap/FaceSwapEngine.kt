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

import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo


class FaceSwapEngine(
    private val context: Context
) {

    companion object {

        private const val TAG =
            "ProFaceSwapEngine"

        private const val ARC_SIZE =
            112

        private const val HYPER_SIZE =
            256

        /*
         * Target-face ROI padding.
         *
         * OpenCV processing is performed only inside this
         * local region instead of the complete photograph.
         */
        private const val ROI_PADDING_FACTOR =
            0.32f

        private const val MIN_ROI_PADDING =
            24f

        /*
         * Prevent unnecessarily huge OpenCV regions.
         */
        private const val MAX_ROI_SIZE =
            1200

        /*
         * Target landmark mask expansion.
         */
        private const val MASK_EXPANSION =
            1.02f
    }


    private val environment =
        OrtEnvironment.getEnvironment()


    private var detectorSession:
        OrtSession? = null

    private var landmarkerSession:
        OrtSession? = null

    private var arcFaceSession:
        OrtSession? = null

    private var hyperSwapSession:
        OrtSession? = null


    /*
     * ============================================================
     * MODEL LOADING
     * ============================================================
     */

    fun loadModels(): Boolean {

        return try {

            if (!OpenCVLoader.initLocal()) {
                throw IllegalStateException(
                    "OpenCV initialization failed"
                )
            }

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
                "Loading 478-point Face Landmarker"
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


    /*
     * ============================================================
     * MAIN FACE SWAP
     *
     * target = image whose face location is replaced
     * source = image providing the identity
     * ============================================================
     */

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


        /*
         * --------------------------------------------------------
         * SOURCE FACE
         * --------------------------------------------------------
         */

        android.util.Log.d(
            TAG,
            "Detecting SOURCE face"
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


        /*
         * --------------------------------------------------------
         * TARGET FACE
         * --------------------------------------------------------
         */

        android.util.Log.d(
            TAG,
            "Detecting TARGET face"
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


        /*
         * Largest face is used in each image.
         */

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
            "SOURCE face = " +
                    "${sourceFace.left}, " +
                    "${sourceFace.top}, " +
                    "${sourceFace.right}, " +
                    "${sourceFace.bottom}"
        )

        android.util.Log.d(
            TAG,
            "TARGET face = " +
                    "${targetFace.left}, " +
                    "${targetFace.top}, " +
                    "${targetFace.right}, " +
                    "${targetFace.bottom}"
        )


        /*
         * --------------------------------------------------------
         * PREPARE SOURCE
         * --------------------------------------------------------
         */

        val sourceAligned =
            prepareFace(
                source,
                sourceFace,
                landmarker
            )


        /*
         * --------------------------------------------------------
         * PREPARE TARGET
         * --------------------------------------------------------
         */

        val targetAligned =
            prepareFace(
                target,
                targetFace,
                landmarker
            )


        try {

            /*
             * ----------------------------------------------------
             * ARC FACE
             *
             * SOURCE image provides identity.
             * ----------------------------------------------------
             */

            android.util.Log.d(
                TAG,
                "Creating SOURCE ArcFace embedding"
            )

            val sourceEmbedding =
                createArcFaceEmbedding(
                    sourceAligned.bitmap,
                    arcFace
                )


            /*
             * ----------------------------------------------------
             * HYPERSWAP
             *
             * TARGET aligned face is the face being replaced.
             * ----------------------------------------------------
             */

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


            /*
             * ----------------------------------------------------
             * PASTE / BLEND
             * ----------------------------------------------------
             */

            android.util.Log.d(
                TAG,
                "Pasting swapped face at TARGET location"
            )

            return pasteBack(
                original = target,
                swap = swap,
                transform = targetAligned.transform,
                targetLandmarks =
                    targetAligned.landmarks
            )

        } finally {

            if (!sourceAligned.bitmap.isRecycled) {
                sourceAligned.bitmap.recycle()
            }

            if (!targetAligned.bitmap.isRecycled) {
                targetAligned.bitmap.recycle()
            }
        }
    }


    /*
     * ============================================================
     * FACE PREPARATION
     * ============================================================
     */

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


        /*
         * 1.5x face ROI.
         */

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


        /*
         * Keep ROI inside image.
         */

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


        /*
         * Face Landmarker operates on the crop.
         */

        val rawLandmarks =
            faceLandmarker.detect(
                crop
            )


        /*
         * MediaPipe model is expected to provide
         * the full landmark set.
         */

        if (
            rawLandmarks.size < 292
        ) {

            crop.recycle()

            throw IllegalStateException(
                "Could not obtain 478 face landmarks"
            )
        }


        /*
         * Convert landmarks from 256x256 landmarker
         * coordinates back into original image coordinates.
         */

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
                        rawLandmarks[index].size > 2
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


        /*
         * Similarity alignment into HyperSwap's
         * 256x256 face template.
         */

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


        return PreparedFace(
            bitmap = aligned,
            transform = matrix,
            landmarks = landmarks
        )
    }


    /*
     * ============================================================
     * HYPERSWAP TEMPLATE
     * ============================================================
     */

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


    /*
     * ============================================================
     * WARP
     * ============================================================
     */

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
            Canvas(output)

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


    /*
     * ============================================================
     * SIMILARITY TRANSFORM
     * ============================================================
     */

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

                (
                    scale * cos
                    ).toFloat(),

                (
                    -scale * sin
                    ).toFloat(),

                translateX.toFloat(),


                (
                    scale * sin
                    ).toFloat(),

                (
                    scale * cos
                    ).toFloat(),

                translateY.toFloat(),


                0f,
                0f,
                1f
            )
        )


        return matrix
    }


    /*
     * ============================================================
     * ARCFACE
     * ============================================================
     */

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
                            (
                                pixel shr 16
                                ) and 0xFF

                        1 ->
                            (
                                pixel shr 8
                                ) and 0xFF

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


        try {

            val inputName =
                session.inputNames.first()


            val result =
                session.run(
                    mapOf(
                        inputName to
                                tensor
                    )
                )


            try {

                val embedding =
                    extractFloatArray(
                        result[0].value
                    )


                if (
                    embedding.size != 512
                ) {

                    throw IllegalStateException(
                        "ArcFace returned " +
                                "${embedding.size} values"
                    )
                }


                return normalizeEmbedding(
                    embedding
                )

            } finally {

                result.close()
            }

        } finally {

            tensor.close()

            if (!resized.isRecycled) {
                resized.recycle()
            }
        }
    }


    /*
     * ============================================================
     * HYPERSWAP
     * ============================================================
     */

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
                            (
                                pixel shr 16
                                ) and 0xFF

                        1 ->
                            (
                                pixel shr 8
                                ) and 0xFF

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


        var result:
            OrtSession.Result? = null


        try {

            val inputNames =
                session.inputNames.toList()


            if (inputNames.size < 2) {

                throw IllegalStateException(
                    "HyperSwap expects two inputs: " +
                            "$inputNames"
                )
            }


            val inputMap =
                HashMap<String, OnnxTensor>()


            /*
             * HyperSwap 1a:
             *
             * source = ArcFace identity
             * target = target face image
             */

            if (
                inputNames.contains(
                    "source"
                )
            ) {

                inputMap[
                    "source"
                ] =
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

                inputMap[
                    "target"
                ] =
                    targetTensor

            } else {

                inputMap[
                    inputNames[1]
                ] =
                    targetTensor
            }


            android.util.Log.d(
                TAG,
                "HyperSwap inputs = $inputNames"
            )


            result =
                try {

                    session.run(
                        inputMap
                    )

                } catch (e: Throwable) {

                    throw IllegalStateException(
                        "HyperSwap inference failed: " +
                                e.message,
                        e
                    )
                }


            val resultCount =
                result.size()


            if (resultCount < 2) {

                throw IllegalStateException(
                    "HyperSwap returned only " +
                            "$resultCount outputs"
                )
            }


            val imageRaw =
                result[0].value


            val maskRaw =
                result[1].value


            val image =
                extractFloatArray(
                    imageRaw
                )


            val expectedImage =
                3 *
                        HYPER_SIZE *
                        HYPER_SIZE


            if (
                image.size !=
                expectedImage
            ) {

                throw IllegalStateException(
                    "Invalid HyperSwap image output: " +
                            "expected $expectedImage, " +
                            "got ${image.size}"
                )
            }


            val mask =
                extractFloatArray(
                    maskRaw
                )


            val expectedMask =
                HYPER_SIZE *
                        HYPER_SIZE


            if (
                mask.size <
                expectedMask
            ) {

                throw IllegalStateException(
                    "Invalid HyperSwap mask output: " +
                            "expected at least $expectedMask, " +
                            "got ${mask.size}"
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

            try {
                result?.close()
            } catch (_: Throwable) {
            }

            sourceTensor.close()
            targetTensor.close()
        }
    }


    /*
     * ============================================================
     * PASTE BACK
     *
     * First create a safe direct result.
     *
     * Then try ROI-only OpenCV seamlessClone.
     *
     * If OpenCV fails for any reason, the direct result is kept.
     * ============================================================
     */

    private fun pasteBack(
        original: Bitmap,
        swap: HyperSwapResult,
        transform: Matrix,
        targetLandmarks: Array<FloatArray>
    ): Bitmap {

        val inverse =
            Matrix()


        if (
            !transform.invert(
                inverse
            )
        ) {

            swap.bitmap.recycle()

            throw IllegalStateException(
                "Could not invert target face transform"
            )
        }


        /*
         * HyperSwap mask + target landmark mask.
         */

        val maskedSwap =
            createMaskedSwap(
                bitmap = swap.bitmap,
                mask = swap.mask,
                targetLandmarks = targetLandmarks,
                transform = transform
            )


        /*
         * --------------------------------------------------------
         * SAFETY BASE RESULT
         *
         * This guarantees that the target face is placed at the
         * actual target-face location even if OpenCV fails.
         * --------------------------------------------------------
         */

        val directOutput =
            original.copy(
                Bitmap.Config.ARGB_8888,
                true
            )


        val directCanvas =
            Canvas(
                directOutput
            )


        val directPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG or
                        Paint.DITHER_FLAG
            )


        directCanvas.drawBitmap(
            maskedSwap,
            inverse,
            directPaint
        )


        /*
         * --------------------------------------------------------
         * TRY LOCAL ROI SEAMLESS CLONE
         * --------------------------------------------------------
         */

        val blended =
            try {

                android.util.Log.d(
                    TAG,
                    "Trying ROI-only OpenCV seamlessClone"
                )


                val result =
                    seamlessCloneTargetRoi(
                        originalTarget = original,
                        maskedSwap = maskedSwap,
                        inverseTransform = inverse,
                        targetLandmarks = targetLandmarks
                    )


                if (result != null) {

                    android.util.Log.d(
                        TAG,
                        "ROI seamlessClone succeeded"
                    )

                    result

                } else {

                    android.util.Log.w(
                        TAG,
                        "ROI seamlessClone returned null"
                    )

                    directOutput
                }

            } catch (e: Throwable) {

                android.util.Log.w(
                    TAG,
                    "ROI seamlessClone failed; " +
                            "using safe direct paste",
                    e
                )

                directOutput
            }


        /*
         * If the OpenCV result is different from directOutput,
         * release directOutput.
         */

        if (
            blended !== directOutput &&
            !directOutput.isRecycled
        ) {

            directOutput.recycle()
        }


        if (
            !maskedSwap.isRecycled
        ) {

            maskedSwap.recycle()
        }


        if (
            !swap.bitmap.isRecycled
        ) {

            swap.bitmap.recycle()
        }


        return blended
    }


    /*
     * ============================================================
     * ROI SEAMLESS CLONE
     *
     * IMPORTANT:
     *
     * OpenCV never receives the complete photograph.
     *
     * Only the target-face ROI is converted to Mats.
     * ============================================================
     */

    private fun seamlessCloneTargetRoi(
        originalTarget: Bitmap,
        maskedSwap: Bitmap,
        inverseTransform: Matrix,
        targetLandmarks: Array<FloatArray>
    ): Bitmap? {

        if (
            targetLandmarks.size < 3
        ) {
            return null
        }


        /*
         * --------------------------------------------------------
         * FIND TARGET FACE BOUNDS
         * --------------------------------------------------------
         */

        var minX =
            Float.POSITIVE_INFINITY

        var minY =
            Float.POSITIVE_INFINITY

        var maxX =
            Float.NEGATIVE_INFINITY

        var maxY =
            Float.NEGATIVE_INFINITY


        for (point in targetLandmarks) {

            if (point.size < 2) {
                continue
            }

            val x =
                point[0]

            val y =
                point[1]


            if (
                !x.isFinite() ||
                !y.isFinite()
            ) {
                continue
            }


            minX =
                minOf(
                    minX,
                    x
                )

            minY =
                minOf(
                    minY,
                    y
                )

            maxX =
                maxOf(
                    maxX,
                    x
                )

            maxY =
                maxOf(
                    maxY,
                    y
                )
        }


        if (
            !minX.isFinite() ||
            !minY.isFinite() ||
            !maxX.isFinite() ||
            !maxY.isFinite()
        ) {
            return null
        }


        val faceWidth =
            maxX -
                    minX

        val faceHeight =
            maxY -
                    minY


        if (
            faceWidth < 8f ||
            faceHeight < 8f
        ) {
            return null
        }


        /*
         * Add padding around the target face.
         *
         * This is important for seamlessClone because
         * OpenCV needs some surrounding target pixels.
         */

        val padding =
            max(
                MIN_ROI_PADDING,
                max(
                    faceWidth,
                    faceHeight
                ) *
                        ROI_PADDING_FACTOR
            )


        var roiLeft =
            (minX - padding)
                .toInt()

        var roiTop =
            (minY - padding)
                .toInt()

        var roiRight =
            (maxX + padding)
                .toInt()

        var roiBottom =
            (maxY + padding)
                .toInt()


        roiLeft =
            roiLeft.coerceIn(
                0,
                originalTarget.width - 1
            )

        roiTop =
            roiTop.coerceIn(
                0,
                originalTarget.height - 1
            )

        roiRight =
            roiRight.coerceIn(
                roiLeft + 1,
                originalTarget.width
            )

        roiBottom =
            roiBottom.coerceIn(
                roiTop + 1,
                originalTarget.height
            )


        /*
         * Limit ROI dimensions.
         *
         * This keeps OpenCV local and RAM-safe.
         */

        var roiWidth =
            roiRight -
                    roiLeft

        var roiHeight =
            roiBottom -
                    roiTop


        if (
            roiWidth > MAX_ROI_SIZE
        ) {

            val center =
                (
                    roiLeft +
                            roiRight
                    ) * 0.5f

            roiWidth =
                MAX_ROI_SIZE

            roiLeft =
                (
                    center -
                            roiWidth * 0.5f
                    ).toInt()

            roiLeft =
                roiLeft.coerceIn(
                    0,
                    originalTarget.width -
                            roiWidth
                )

            roiRight =
                roiLeft +
                        roiWidth
        }


        if (
            roiHeight > MAX_ROI_SIZE
        ) {

            val center =
                (
                    roiTop +
                            roiBottom
                    ) * 0.5f

            roiHeight =
                MAX_ROI_SIZE

            roiTop =
                (
                    center -
                            roiHeight * 0.5f
                    ).toInt()

            roiTop =
                roiTop.coerceIn(
                    0,
                    originalTarget.height -
                            roiHeight
                )

            roiBottom =
                roiTop +
                        roiHeight
        }


        if (
            roiWidth < 16 ||
            roiHeight < 16
        ) {
            return null
        }


        android.util.Log.d(
            TAG,
            "OpenCV target ROI = " +
                    "$roiLeft,$roiTop " +
                    "${roiWidth}x${roiHeight}"
        )


        /*
         * --------------------------------------------------------
         * TARGET ROI
         * --------------------------------------------------------
         */

        val targetRoiBitmap =
            Bitmap.createBitmap(
                originalTarget,
                roiLeft,
                roiTop,
                roiWidth,
                roiHeight
            )


        /*
         * --------------------------------------------------------
         * SOURCE ROI
         *
         * Render the already-masked HyperSwap image into the
         * same original-image ROI coordinate system.
         * --------------------------------------------------------
         */

        val sourceRoiBitmap =
            Bitmap.createBitmap(
                roiWidth,
                roiHeight,
                Bitmap.Config.ARGB_8888
            )


        val sourceCanvas =
            Canvas(
                sourceRoiBitmap
            )


        sourceCanvas.drawColor(
            Color.BLACK
        )


        /*
         * Convert global inverse transform into ROI-local
         * coordinates.
         *
         * This is critical:
         *
         * destination = original coordinates - ROI origin
         */

        val localMatrix =
            Matrix()


        localMatrix.set(
            inverseTransform
        )


        val matrixValues =
            FloatArray(
                9
            )


        localMatrix.getValues(
            matrixValues
        )


        matrixValues[Matrix.MTRANS_X] -=
            roiLeft.toFloat()

        matrixValues[Matrix.MTRANS_Y] -=
            roiTop.toFloat()


        localMatrix.setValues(
            matrixValues
        )


        val sourcePaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG or
                        Paint.DITHER_FLAG
            )


        sourceCanvas.drawBitmap(
            maskedSwap,
            localMatrix,
            sourcePaint
        )


        /*
         * --------------------------------------------------------
         * CONVERT TO OPENCV
         * --------------------------------------------------------
         */

        val sourceRgba =
            Mat()

        val targetRgba =
            Mat()

        val sourceBgr =
            Mat()

        val targetBgr =
            Mat()

        val alpha =
            Mat()

        val faceMask =
            Mat()

        val combinedMask =
            Mat()

        val blendBgr =
            Mat()

        val blendRgba =
            Mat()


        try {

            Utils.bitmapToMat(
                sourceRoiBitmap,
                sourceRgba
            )

            Utils.bitmapToMat(
                targetRoiBitmap,
                targetRgba
            )


            /*
             * Android Bitmap -> OpenCV BGR.
             */

            Imgproc.cvtColor(
                sourceRgba,
                sourceBgr,
                Imgproc.COLOR_RGBA2BGR
            )

            Imgproc.cvtColor(
                targetRgba,
                targetBgr,
                Imgproc.COLOR_RGBA2BGR
            )


            /*
             * Extract alpha from rendered HyperSwap.
             *
             * This prevents black background pixels from
             * being cloned outside the actual generated face.
             */

            Core.extractChannel(
                sourceRgba,
                alpha,
                3
            )


            /*
             * ----------------------------------------------------
             * TARGET LANDMARK HULL
             * ----------------------------------------------------
             */

            val localPoints =
                ArrayList<Point>()


            for (point in targetLandmarks) {

                if (point.size < 2) {
                    continue
                }

                val x =
                    point[0]

                val y =
                    point[1]


                if (
                    !x.isFinite() ||
                    !y.isFinite()
                ) {
                    continue
                }


                val localX =
                    (
                        x -
                                roiLeft
                        ).coerceIn(
                            0f,
                            (
                                roiWidth -
                                        1
                                ).toFloat()
                        )

                val localY =
                    (
                        y -
                                roiTop
                        ).coerceIn(
                            0f,
                            (
                                roiHeight -
                                        1
                                ).toFloat()
                        )


                localPoints.add(
                    Point(
                        localX.toDouble(),
                        localY.toDouble()
                    )
                )
            }


            if (
                localPoints.size < 3
            ) {
                return null
            }


            /*
             * Convex hull.
             */

            val hullPoints =
                convexHullOpenCv(
                    localPoints
                )


            if (
                hullPoints.size < 3
            ) {
                return null
            }


            /*
             * Slight expansion.
             *
             * Target geometry controls where blending happens.
             */

            val centerX =
                hullPoints
                    .map {
                        it.x
                    }
                    .average()

            val centerY =
                hullPoints
                    .map {
                        it.y
                    }
                    .average()


            val expandedPoints =
                hullPoints.map {

                    val dx =
                        it.x -
                                centerX

                    val dy =
                        it.y -
                                centerY

                    Point(
                        centerX +
                                dx *
                                MASK_EXPANSION,

                        centerY +
                                dy *
                                MASK_EXPANSION
                    )
                }


            val hullMat =
                MatOfPoint(
                    *expandedPoints.map {
                        Point(
                            it.x,
                            it.y
                        )
                    }.toTypedArray()
                )


            /*
             * Binary target-face mask.
             */

            faceMask.create(
                roiHeight,
                roiWidth,
                CvType.CV_8UC1
            )

            faceMask.setTo(
                Scalar(
                    0.0
                )
            )


            Imgproc.fillConvexPoly(
                faceMask,
                hullMat,
                Scalar(
                    255.0
                )
            )


            /*
             * Small blur on the mask edge.
             *
             * OpenCV seamlessClone treats non-zero mask pixels
             * as the selected source region.
             */

            Imgproc.GaussianBlur(
                faceMask,
                faceMask,
                Size(
                    9.0,
                    9.0
                ),
                0.0
            )


            /*
             * Make sure there is no completely invisible
             * source region being cloned.
             *
             * Combine:
             *
             * TARGET landmark mask
             * +
             * HyperSwap rendered alpha
             */

            Core.min(
                faceMask,
                alpha,
                combinedMask
            )


            /*
             * Threshold to a usable non-zero mask.
             */

            Imgproc.threshold(
                combinedMask,
                combinedMask,
                8.0,
                255.0,
                Imgproc.THRESH_BINARY
            )


            /*
             * ----------------------------------------------------
             * SAFETY CHECK
             *
             * OpenCV requires an 8-bit 3-channel source/destination
             * for seamlessClone.
             * ----------------------------------------------------
             */

            if (
                sourceBgr.type() !=
                CvType.CV_8UC3
            ) {

                throw IllegalStateException(
                    "Invalid source ROI type: " +
                            sourceBgr.type()
                )
            }


            if (
                targetBgr.type() !=
                CvType.CV_8UC3
            ) {

                throw IllegalStateException(
                    "Invalid target ROI type: " +
                            targetBgr.type()
                )
            }


            /*
             * ----------------------------------------------------
             * SEAMLESS CLONE
             *
             * The point is the CENTER OF THIS ROI.
             *
             * It is NOT the center of the original photograph.
             * ----------------------------------------------------
             */

            val cloneCenter =
                Point(
                    roiWidth / 2.0,
                    roiHeight / 2.0
                )


            Photo.seamlessClone(
                sourceBgr,
                targetBgr,
                combinedMask,
                cloneCenter,
                blendBgr,
                Photo.MIXED_CLONE
            )


            /*
             * Convert result back to Android RGBA.
             */

            Imgproc.cvtColor(
                blendBgr,
                blendRgba,
                Imgproc.COLOR_BGR2RGBA
            )


            val blendedRoiBitmap =
                Bitmap.createBitmap(
                    roiWidth,
                    roiHeight,
                    Bitmap.Config.ARGB_8888
                )


            Utils.matToBitmap(
                blendRgba,
                blendedRoiBitmap
            )


            /*
             * ----------------------------------------------------
             * PUT ONLY THE ROI BACK INTO THE ORIGINAL IMAGE
             * ----------------------------------------------------
             */

            val output =
                originalTarget.copy(
                    Bitmap.Config.ARGB_8888,
                    true
                )


            val outputCanvas =
                Canvas(
                    output
                )


            outputCanvas.drawBitmap(
                blendedRoiBitmap,
                roiLeft.toFloat(),
                roiTop.toFloat(),
                Paint(
                    Paint.ANTI_ALIAS_FLAG or
                            Paint.FILTER_BITMAP_FLAG or
                            Paint.DITHER_FLAG
                )
            )


            blendedRoiBitmap.recycle()


            return output

        } finally {

            /*
             * Release every local OpenCV Mat immediately.
             */

            sourceRgba.release()
            targetRgba.release()

            sourceBgr.release()
            targetBgr.release()

            alpha.release()
            faceMask.release()
            combinedMask.release()

            blendBgr.release()
            blendRgba.release()
        }


    }


    /*
     * ============================================================
     * MASKED HYPERSWAP RESULT
     *
     * HyperSwap mask remains the primary mask.
     *
     * Target landmarks are used to prevent the result from
     * wandering outside the target face.
     * ============================================================
     */

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


        /*
         * Build target landmark mask in HyperSwap coordinates.
         */

        val targetMask =
            createTargetFaceMask(
                targetLandmarks,
                transform,
                HYPER_SIZE,
                HYPER_SIZE
            )


        val targetMaskPixels =
            ByteArray(
                expected
            )


        targetMask.copyPixelsToBuffer(
            java.nio.ByteBuffer.wrap(
                targetMaskPixels
            )
        )


        targetMask.recycle()


        val outputPixels =
            IntArray(
                expected
            )


        for (i in 0 until expected) {

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
                        .toInt() and
                            0xFF
                    ) / 255f


            /*
             * HyperSwap controls the generated face.
             *
             * Target landmark mask prevents the face from
             * escaping the actual target-face geometry.
             */

            val combinedAlpha =
                (
                    hyperAlpha +
                            landmarkAlpha *
                            0.65f *
                            (
                                1f -
                                        hyperAlpha
                                )
                    )
                    .coerceIn(
                        0f,
                        0.97f
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


    /*
     * ============================================================
     * TARGET LANDMARK MASK
     * ============================================================
     */

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
            Canvas(
                mask
            )


        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )


        paint.color =
            Color.WHITE

        paint.style =
            Paint.Style.FILL


        val points =
            FloatArray(
                targetLandmarks.size *
                        2
            )


        for (
            i in targetLandmarks.indices
        ) {

            points[i * 2] =
                targetLandmarks[i][0]

            points[i * 2 + 1] =
                targetLandmarks[i][1]
        }


        /*
         * Original target coordinates ->
         * HyperSwap 256x256 coordinates.
         */

        transform.mapPoints(
            points
        )


        val hull =
            convexHullPoints(
                points
            )


        if (
            hull.size < 3
        ) {

            return mask
        }


        val centerX =
            hull
                .map {
                    it.x
                }
                .average()
                .toFloat()


        val centerY =
            hull
                .map {
                    it.y
                }
                .average()
                .toFloat()


        /*
         * Slight shrink so the landmark boundary does not
         * extend too far into hair/background.
         */

        val shrink =
            0.96f


        for (point in hull) {

            point.x =
                centerX +
                        (
                            point.x -
                                    centerX
                            ) *
                        shrink

            point.y =
                centerY +
                        (
                            point.y -
                                    centerY
                            ) *
                        shrink
        }


        val path =
            Path()


        path.moveTo(
            hull[0].x,
            hull[0].y
        )


        for (
            i in 1 until hull.size
        ) {

            path.lineTo(
                hull[i].x,
                hull[i].y
            )
        }


        path.close()


        /*
         * Solid face region.
         */

        canvas.drawPath(
            path,
            paint
        )


        /*
         * Small feather.
         */

        val featherPaint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )


        featherPaint.color =
            Color.WHITE

        featherPaint.style =
            Paint.Style.FILL


        featherPaint.maskFilter =
            android.graphics.BlurMaskFilter(
                8f,
                android.graphics.BlurMaskFilter.Blur.NORMAL
            )


        canvas.drawPath(
            path,
            featherPaint
        )


        return mask
    }


    /*
     * ============================================================
     * OPENCV CONVEX HULL
     * ============================================================
     */

    private fun convexHullOpenCv(
        points: List<Point>
    ): List<Point> {

        if (
            points.size <= 2
        ) {
            return points
        }


        val sorted =
            points.sortedWith(
                compareBy<Point> {
                    it.x
                }.thenBy {
                    it.y
                }
            )


        fun cross(
            a: Point,
            b: Point,
            c: Point
        ): Double {

            return (
                b.x - a.x
                ) *
                    (
                        c.y - a.y
                        ) -
                    (
                        b.y - a.y
                        ) *
                    (
                        c.x - a.x
                        )
        }


        val lower =
            ArrayList<Point>()


        for (point in sorted) {

            while (
                lower.size >= 2 &&
                cross(
                    lower[
                        lower.size - 2
                    ],
                    lower[
                        lower.size - 1
                    ],
                    point
                ) <= 0.0
            ) {

                lower.removeAt(
                    lower.size - 1
                )
            }


            lower.add(
                point
            )
        }


        val upper =
            ArrayList<Point>()


        for (
            index in
            sorted.indices.reversed()
        ) {

            val point =
                sorted[index]


            while (
                upper.size >= 2 &&
                cross(
                    upper[
                        upper.size - 2
                    ],
                    upper[
                        upper.size - 1
                    ],
                    point
                ) <= 0.0
            ) {

                upper.removeAt(
                    upper.size - 1
                )
            }


            upper.add(
                point
            )
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


    /*
     * ============================================================
     * BITMAP DECODER
     * ============================================================
     */

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


        for (i in 0 until plane) {

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
                    values[
                        plane + i
                    ] *
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
                    values[
                        plane * 2 + i
                    ] *
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


    /*
     * ============================================================
     * TENSOR DECODER
     * ============================================================
     */

    private fun extractFloatArray(
        raw: Any
    ): FloatArray {

        fun flatten(
            value: Any?
        ): MutableList<Float> {

            val output =
                mutableListOf<Float>()


            when (value) {

                is FloatArray -> {

                    output.addAll(
                        value.toList()
                    )
                }


                is DoubleArray -> {

                    output.addAll(
                        value.map {
                            it.toFloat()
                        }
                    )
                }


                is Array<*> -> {

                    value.forEach {

                        output.addAll(
                            flatten(
                                it
                            )
                        )
                    }
                }


                null -> {
                }


                else -> {

                    throw IllegalStateException(
                        "Unexpected tensor type: " +
                                value.javaClass.name
                    )
                }
            }


            return output
        }


        return flatten(
            raw
        ).toFloatArray()
    }


    /*
     * ============================================================
     * NORMALIZE ARCFACE EMBEDDING
     * ============================================================
     */

    private fun normalizeEmbedding(
        values: FloatArray
    ): FloatArray {

        var sum =
            0.0


        for (value in values) {

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


    /*
     * ============================================================
     * LARGEST FACE
     * ============================================================
     */

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


    /*
     * ============================================================
     * MODEL SESSION
     * ============================================================
     */

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


    /*
     * ============================================================
     * CLOSE
     * ============================================================
     */

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


    /*
     * ============================================================
     * ANDROID CONVEX HULL
     * ============================================================
     */

    private fun convexHullPoints(
        points: FloatArray
    ): List<PointF> {

        val input =
            ArrayList<PointF>()


        var i = 0


        while (
            i + 1 < points.size
        ) {

            input.add(
                PointF(
                    points[i],
                    points[i + 1]
                )
            )

            i += 2
        }


        if (
            input.size <= 2
        ) {

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
                b.x - a.x
                ) *
                    (
                        c.y - a.y
                        ) -
                    (
                        b.y - a.y
                        ) *
                    (
                        c.x - a.x
                        )
        }


        val lower =
            ArrayList<PointF>()


        for (point in sorted) {

            while (
                lower.size >= 2 &&
                cross(
                    lower[
                        lower.size - 2
                    ],
                    lower[
                        lower.size - 1
                    ],
                    point
                ) <= 0f
            ) {

                lower.removeAt(
                    lower.size - 1
                )
            }


            lower.add(
                point
            )
        }


        val upper =
            ArrayList<PointF>()


        for (
            index in
            sorted.indices.reversed()
        ) {

            val point =
                sorted[index]


            while (
                upper.size >= 2 &&
                cross(
                    upper[
                        upper.size - 2
                    ],
                    upper[
                        upper.size - 1
                    ],
                    point
                ) <= 0f
            ) {

                upper.removeAt(
                    upper.size - 1
                )
            }


            upper.add(
                point
            )
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


    /*
     * ============================================================
     * DATA CLASSES
     * ============================================================
     */

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
