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
         * Keep OpenCV local.
         * Never process the complete photograph.
         */
        private const val MAX_ROI_SIZE =
            768

        /*
         * Smaller ROI helps prevent ghosting and
         * reduces native memory usage.
         */
        private const val ROI_PADDING_FACTOR =
            0.22f

        private const val MIN_ROI_PADDING =
            18f

        /*
         * Higher threshold = tighter HyperSwap mask.
         */
        private const val MASK_THRESHOLD =
            0.12f
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
     * LOAD MODELS
     * ============================================================
     */

    fun loadModels(): Boolean {

        return try {

            if (!OpenCVLoader.initLocal()) {

                throw IllegalStateException(
                    "OpenCV initialization failed"
                )
            }

            detectorSession =
                createSession(
                    "models/face_detection_short_range.onnx"
                )

            landmarkerSession =
                createSession(
                    "models/face_landmarker_Nx3x256x256.onnx"
                )

            arcFaceSession =
                createSession(
                    "models/arcface_w600k_r50.onnx"
                )

            android.util.Log.d(
                TAG,
                "Base models loaded"
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

            if (
                hyperSwapSession != null
            ) {
                return true
            }

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
     * MAIN SWAP
     *
     * source = identity
     * target = location
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

        if (
            !loadHyperSwap()
        ) {

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
         * SOURCE
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

        if (
            sourceFaces.isEmpty()
        ) {

            throw IllegalStateException(
                "No face found in source image"
            )
        }


        /*
         * --------------------------------------------------------
         * TARGET
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

        if (
            targetFaces.isEmpty()
        ) {

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
            "SOURCE face detected"
        )

        android.util.Log.d(
            TAG,
            "TARGET face detected"
        )


        /*
         * --------------------------------------------------------
         * ALIGN
         * --------------------------------------------------------
         */

        val sourceAligned =
            prepareFace(
                source,
                sourceFace,
                landmarker
            )

        val targetAligned =
            prepareFace(
                target,
                targetFace,
                landmarker
            )


        try {

            /*
             * SOURCE provides identity.
             */

            val sourceEmbedding =
                createArcFaceEmbedding(
                    sourceAligned.bitmap,
                    arcFace
                )


            /*
             * HyperSwap replaces TARGET face.
             */

            val swap =
                runHyperSwap(
                    sourceEmbedding,
                    targetAligned.bitmap,
                    hyperSwap
                )


            /*
             * Put result back at TARGET location.
             */

            return pasteBack(
                original =
                    target,

                swap =
                    swap,

                transform =
                    targetAligned.transform,

                targetLandmarks =
                    targetAligned.landmarks
            )

        } finally {

            recycleBitmap(
                sourceAligned.bitmap
            )

            recycleBitmap(
                targetAligned.bitmap
            )
        }
    }


    /*
     * ============================================================
     * PREPARE FACE
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


        val faceWidth =
            detection.right -
                    detection.left

        val faceHeight =
            detection.bottom -
                    detection.top

        val faceSize =
            max(
                faceWidth,
                faceHeight
            )


        if (
            faceSize <= 1f
        ) {

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


        var left =
            centerX -
                    roiSize * 0.5f

        var top =
            centerY -
                    roiSize * 0.5f

        var right =
            centerX +
                    roiSize * 0.5f

        var bottom =
            centerY +
                    roiSize * 0.5f


        if (
            left < 0f
        ) {

            val d =
                -left

            left = 0f
            right += d
        }


        if (
            top < 0f
        ) {

            val d =
                -top

            top = 0f
            bottom += d
        }


        if (
            right >
            bitmap.width.toFloat()
        ) {

            val d =
                right -
                        bitmap.width.toFloat()

            right =
                bitmap.width.toFloat()

            left -= d
        }


        if (
            bottom >
            bitmap.height.toFloat()
        ) {

            val d =
                bottom -
                        bitmap.height.toFloat()

            bottom =
                bitmap.height.toFloat()

            top -= d
        }


        left =
            left.coerceIn(
                0f,
                bitmap.width.toFloat() - 1f
            )

        top =
            top.coerceIn(
                0f,
                bitmap.height.toFloat() - 1f
            )

        right =
            right.coerceIn(
                left + 1f,
                bitmap.width.toFloat()
            )

        bottom =
            bottom.coerceIn(
                top + 1f,
                bitmap.height.toFloat()
            )


        val cropLeft =
            left.toInt()

        val cropTop =
            top.toInt()

        val cropWidth =
            max(
                1,
                (right - left).toInt()
            ).coerceAtMost(
                bitmap.width -
                        cropLeft
            )

        val cropHeight =
            max(
                1,
                (bottom - top).toInt()
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
            try {

                faceLandmarker.detect(
                    crop
                )

            } finally {
            }


        if (
            rawLandmarks.size < 292
        ) {

            recycleBitmap(
                crop
            )

            throw IllegalStateException(
                "Could not obtain face landmarks"
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
                        rawLandmarks[index].size > 2
                    ) {

                        rawLandmarks[index][2]

                    } else {

                        0f
                    }
                )
            }


        for (
            point in landmarks
        ) {

            point[0] +=
                cropLeft.toFloat()

            point[1] +=
                cropTop.toFloat()
        }


        recycleBitmap(
            crop
        )


        val matrix =
            calculateSimilarityTransform(
                FaceAlignment.fivePoints(
                    landmarks
                ),
                hyperSwapTemplate()
            )


        val aligned =
            warpBitmap(
                bitmap,
                matrix,
                HYPER_SIZE
            )


        return PreparedFace(
            bitmap =
                aligned,

            transform =
                matrix,

            landmarks =
                landmarks
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


    /*
     * ============================================================
     * SIMILARITY TRANSFORM
     * ============================================================
     */

    private fun calculateSimilarityTransform(
        source: Array<FloatArray>,
        target: Array<FloatArray>
    ): Matrix {

        var sourceCx =
            0.0

        var sourceCy =
            0.0

        var targetCx =
            0.0

        var targetCy =
            0.0


        for (
            i in 0 until 5
        ) {

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


        var a =
            0.0

        var b =
            0.0

        var denominator =
            0.0


        for (
            i in 0 until 5
        ) {

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


        try {

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


            for (
                channel in 0..2
            ) {

                for (
                    pixel in pixels
                ) {

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

                } finally {

                    result.close()
                }

            } finally {

                tensor.close()
            }

        } finally {

            recycleBitmap(
                resized
            )
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


        for (
            channel in 0..2
        ) {

            for (
                pixel in pixels
            ) {

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
            OrtSession.Result? =
                null


        try {

            val inputNames =
                session.inputNames.toList()


            if (
                inputNames.size < 2
            ) {

                throw IllegalStateException(
                    "HyperSwap expects two inputs"
                )
            }


            val inputMap =
                HashMap<String, OnnxTensor>()


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


            result =
                session.run(
                    inputMap
                )


            if (
                result.size() < 2
            ) {

                throw IllegalStateException(
                    "HyperSwap returned " +
                            "${result.size()} outputs"
                )
            }


            val image =
                extractFloatArray(
                    result[0].value
                )


            val mask =
                extractFloatArray(
                    result[1].value
                )


            val expectedImage =
                3 *
                        HYPER_SIZE *
                        HYPER_SIZE


            val expectedMask =
                HYPER_SIZE *
                        HYPER_SIZE


            if (
                image.size !=
                expectedImage
            ) {

                throw IllegalStateException(
                    "Invalid HyperSwap image output"
                )
            }


            if (
                mask.size <
                expectedMask
            ) {

                throw IllegalStateException(
                    "Invalid HyperSwap mask output"
                )
            }


            return HyperSwapResult(
                bitmap =
                    imageToBitmap(
                        image
                    ),

                mask =
                    mask
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

            recycleBitmap(
                swap.bitmap
            )

            throw IllegalStateException(
                "Could not invert target transform"
            )
        }


        /*
         * Make the tight masked face.
         */

        val maskedSwap =
            try {

                createMaskedSwap(
                    bitmap =
                        swap.bitmap,

                    mask =
                        swap.mask,

                    targetLandmarks =
                        targetLandmarks,

                    transform =
                        transform
                )

            } finally {

                recycleBitmap(
                    swap.bitmap
                )
            }


        try {

            /*
             * First try high-quality local OpenCV.
             */

            val blended =
                try {

                    seamlessCloneTargetRoi(
                        originalTarget =
                            original,

                        maskedSwap =
                            maskedSwap,

                        inverseTransform =
                            inverse,

                        targetLandmarks =
                            targetLandmarks
                    )

                } catch (e: Throwable) {

                    android.util.Log.e(
                        TAG,
                        "OpenCV ROI failed",
                        e
                    )

                    null
                }


            if (
                blended != null
            ) {

                return blended
            }


            /*
             * Safe fallback.
             */

            return directPaste(
                original =
                    original,

                maskedSwap =
                    maskedSwap,

                inverse =
                    inverse
            )

        } finally {

            recycleBitmap(
                maskedSwap
            )
        }
    }


    /*
     * ============================================================
     * DIRECT TARGET-POSITION FALLBACK
     * ============================================================
     */

    private fun directPaste(
        original: Bitmap,
        maskedSwap: Bitmap,
        inverse: Matrix
    ): Bitmap {

        val output =
            original.copy(
                Bitmap.Config.ARGB_8888,
                true
            )


        val canvas =
            Canvas(
                output
            )


        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG or
                        Paint.DITHER_FLAG
            )


        canvas.drawBitmap(
            maskedSwap,
            inverse,
            paint
        )


        return output
    }


    /*
     * ============================================================
     * ROI SEAMLESS CLONE
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


        for (
            point in targetLandmarks
        ) {

            if (
                point.size < 2
            ) {
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
         * Tight ROI.
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
            (
                minX -
                        padding
                ).toInt()


        var roiTop =
            (
                minY -
                        padding
                ).toInt()


        var roiRight =
            (
                maxX +
                        padding
                ).toInt()


        var roiBottom =
            (
                maxY +
                        padding
                ).toInt()


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


        var roiWidth =
            roiRight -
                    roiLeft


        var roiHeight =
            roiBottom -
                    roiTop


        /*
         * Hard memory limit.
         */

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
            roiWidth < 32 ||
            roiHeight < 32
        ) {

            return null
        }


        /*
         * --------------------------------------------------------
         * CREATE ONLY SMALL ROI BITMAPS
         * --------------------------------------------------------
         */

        val targetRoi =
            Bitmap.createBitmap(
                originalTarget,
                roiLeft,
                roiTop,
                roiWidth,
                roiHeight
            )


        val sourceRoi =
            Bitmap.createBitmap(
                roiWidth,
                roiHeight,
                Bitmap.Config.ARGB_8888
            )


        try {

            /*
             * ----------------------------------------------------
             * RENDER SWAP INTO LOCAL ROI
             * ----------------------------------------------------
             */

            val sourceCanvas =
                Canvas(
                    sourceRoi
                )


            sourceCanvas.drawColor(
                Color.BLACK
            )


            val localMatrix =
                Matrix()


            localMatrix.set(
                inverseTransform
            )


            val values =
                FloatArray(
                    9
                )


            localMatrix.getValues(
                values
            )


            values[
                Matrix.MTRANS_X
            ] -=
                roiLeft.toFloat()


            values[
                Matrix.MTRANS_Y
            ] -=
                roiTop.toFloat()


            localMatrix.setValues(
                values
            )


            sourceCanvas.drawBitmap(
                maskedSwap,
                localMatrix,
                Paint(
                    Paint.ANTI_ALIAS_FLAG or
                            Paint.FILTER_BITMAP_FLAG
                )
            )


            /*
             * ----------------------------------------------------
             * OPENCV
             * ----------------------------------------------------
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

            val mask =
                Mat()

            val cloneResult =
                Mat()

            val cloneRgba =
                Mat()


            try {

                Utils.bitmapToMat(
                    sourceRoi,
                    sourceRgba
                )


                Utils.bitmapToMat(
                    targetRoi,
                    targetRgba
                )


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
                 * Extract HyperSwap alpha.
                 */

                Core.extractChannel(
                    sourceRgba,
                    alpha,
                    3
                )


                /*
                 * Ghost-Fix:
                 *
                 * ONLY use the actual HyperSwap-generated
                 * face mask.
                 *
                 * Do not enlarge it with the entire
                 * 478-point face hull.
                 */

                Imgproc.threshold(
                    alpha,
                    mask,
                    MASK_THRESHOLD * 255.0,
                    255.0,
                    Imgproc.THRESH_BINARY
                )


                /*
                 * Tiny cleanup.
                 */

                val kernel =
                    Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE,
                        Size(
                            3.0,
                            3.0
                        )
                    )


                try {

                    Imgproc.morphologyEx(
                        mask,
                        mask,
                        Imgproc.MORPH_OPEN,
                        kernel
                    )

                } finally {

                    kernel.release()
                }


                /*
                 * Small feather only.
                 */

                Imgproc.GaussianBlur(
                    mask,
                    mask,
                    Size(
                        7.0,
                        7.0
                    ),
                    0.0
                )


                if (
                    Core.countNonZero(
                        mask
                    ) < 50
                ) {

                    return null
                }


                /*
                 * ------------------------------------------------
                 * GHOST FIX:
                 *
                 * NORMAL_CLONE instead of MIXED_CLONE.
                 * ------------------------------------------------
                 */

                val cloneCenter =
                    Point(
                        roiWidth / 2.0,
                        roiHeight / 2.0
                    )


                Photo.seamlessClone(
                    sourceBgr,
                    targetBgr,
                    mask,
                    cloneCenter,
                    cloneResult,
                    Photo.NORMAL_CLONE
                )


                /*
                 * BGR -> RGBA.
                 */

                Imgproc.cvtColor(
                    cloneResult,
                    cloneRgba,
                    Imgproc.COLOR_BGR2RGBA
                )


                val blendedRoi =
                    Bitmap.createBitmap(
                        roiWidth,
                        roiHeight,
                        Bitmap.Config.ARGB_8888
                    )


                try {

                    Utils.matToBitmap(
                        cloneRgba,
                        blendedRoi
                    )


                    /*
                     * ------------------------------------------------
                     * PUT ONLY ROI BACK
                     * ------------------------------------------------
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
                        blendedRoi,
                        roiLeft.toFloat(),
                        roiTop.toFloat(),
                        Paint(
                            Paint.ANTI_ALIAS_FLAG or
                                    Paint.FILTER_BITMAP_FLAG
                        )
                    )


                    return output

                } finally {

                    recycleBitmap(
                        blendedRoi
                    )
                }

            } finally {

                /*
                 * Release ALL native OpenCV memory.
                 */

                sourceRgba.release()
                targetRgba.release()

                sourceBgr.release()
                targetBgr.release()

                alpha.release()
                mask.release()

                cloneResult.release()
                cloneRgba.release()
            }

        } finally {

            /*
             * Release ROI bitmaps immediately.
             */

            recycleBitmap(
                sourceRoi
            )

            recycleBitmap(
                targetRoi
            )
        }
    }


    /*
     * ============================================================
     * MASKED SWAP
     * ============================================================
     */

    private fun createMaskedSwap(
        bitmap: Bitmap,
        mask: FloatArray,
        targetLandmarks: Array<FloatArray>,
        transform: Matrix
    ): Bitmap {

        val total =
            HYPER_SIZE *
                    HYPER_SIZE


        if (
            mask.size < total
        ) {

            throw IllegalStateException(
                "Invalid HyperSwap mask"
            )
        }


        val sourcePixels =
            IntArray(
                total
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


        val outputPixels =
            IntArray(
                total
            )


        /*
         * Tight HyperSwap-only mask.
         */

        for (
            i in 0 until total
        ) {

            val alpha =
                mask[i]
                    .coerceIn(
                        0f,
                        1f
                    )


            val tightened =
                if (
                    alpha <=
                    MASK_THRESHOLD
                ) {

                    0f

                } else {

                    (
                        alpha -
                                MASK_THRESHOLD
                        ) /
                        (
                            1f -
                                    MASK_THRESHOLD
                            )
                }
                    .coerceIn(
                        0f,
                        1f
                    )


            /*
             * Smoothstep alpha.
             */

            val smooth =
                tightened *
                        tightened *
                        (
                            3f -
                                    2f *
                                    tightened
                            )


            val finalAlpha =
                (
                    smooth *
                            255f
                    )
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )


            val color =
                sourcePixels[i]


            outputPixels[i] =
                Color.argb(
                    finalAlpha,
                    Color.red(
                        color
                    ),
                    Color.green(
                        color
                    ),
                    Color.blue(
                        color
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
     * HYPERSWAP OUTPUT -> BITMAP
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
            values.size != expected
        ) {

            throw IllegalStateException(
                "Invalid HyperSwap image output"
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
     * EXTRACT FLOAT ARRAY
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
     * NORMALIZE EMBEDDING
     * ============================================================
     */

    private fun normalizeEmbedding(
        values: FloatArray
    ): FloatArray {

        var sum =
            0.0


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
            norm < 0.000001f
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
     * CREATE MODEL SESSION
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
     * SAFE BITMAP RECYCLE
     * ============================================================
     */

    private fun recycleBitmap(
        bitmap: Bitmap?
    ) {

        try {

            if (
                bitmap != null &&
                !bitmap.isRecycled
            ) {

                bitmap.recycle()
            }

        } catch (_: Throwable) {
        }
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


        hyperSwapSession =
            null

        arcFaceSession =
            null

        landmarkerSession =
            null

        detectorSession =
            null
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
