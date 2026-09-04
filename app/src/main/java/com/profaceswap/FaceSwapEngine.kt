package com.profaceswap

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class FaceSwapEngine(
    private val context: Context
) {

    companion object {

        private const val LANDMARK_MODEL =
            "face_landmarker_Nx3x256x256.onnx"

        private const val DETECTION_MODEL =
            "face_detection_short_range.onnx"

        private const val ARCFACE_MODEL =
            "arcface_w600k_r50.onnx"

        private const val HYPERSWAP_MODEL =
            "hyperswap_1a_256.onnx"

        private const val FACE_SIZE =
            256

        private const val ARCFACE_SIZE =
            112

        /*
         * Adaptive blending limits.
         *
         * These are deliberately moderate.
         * The target landmark hull is the final safety boundary,
         * so the mask cannot expand outside the target face.
         */
        private const val MAX_EXPAND_X =
            22

        private const val MAX_EXPAND_Y =
            20

        private const val MIN_EXPAND =
            2

        private const val MASK_BLUR_RADIUS =
            7

        /*
         * Small inward safety margin.
         * Prevents the final mask from reaching too far into
         * hair/background while still following the target face.
         */
        private const val HULL_SHRINK =
            3.5f

        /*
         * Strength of the final alpha.
         * Keeping this below 1 helps avoid a pasted-on appearance.
         */
        private const val MAX_ALPHA =
            0.97f
    }

    private val environment =
        OrtEnvironment.getEnvironment()

    private var detectorSession: OrtSession? =
        null

    private var landmarkerSession: OrtSession? =
        null

    private var arcFaceSession: OrtSession? =
        null

    private var hyperSwapSession: OrtSession? =
        null

    private var hyperSwapLoaded =
        false

    private var closed =
        false

    fun loadModels(): Boolean {

        return try {

            detectorSession =
                createSession(
                    DETECTION_MODEL
                )

            landmarkerSession =
                createSession(
                    LANDMARK_MODEL
                )

            arcFaceSession =
                createSession(
                    ARCFACE_MODEL
                )

            true

        } catch (e: Throwable) {

            e.printStackTrace()

            false
        }
    }

    private fun loadHyperSwap(): Boolean {

        if (hyperSwapLoaded &&
            hyperSwapSession != null
        ) {
            return true
        }

        return try {

            hyperSwapSession =
                createSession(
                    HYPERSWAP_MODEL
                )

            hyperSwapLoaded =
                true

            true

        } catch (e: Throwable) {

            e.printStackTrace()

            false
        }
    }

    private fun createSession(
        assetName: String
    ): OrtSession {

        val modelBytes =
            context.assets
                .open(assetName)
                .use { input ->
                    input.readBytes()
                }

        return environment.createSession(
            modelBytes,
            OrtSession.SessionOptions()
        )
    }

    fun processSwap(
        target: Bitmap,
        source: Bitmap
    ): Bitmap {

        check(!closed) {
            "FaceSwapEngine is closed"
        }

        if (!loadHyperSwap()) {
            throw IllegalStateException(
                "Could not load HyperSwap model"
            )
        }

        /*
         * Detect faces.
         */
        val sourceFaces =
            detectFaces(source)

        val targetFaces =
            detectFaces(target)

        if (sourceFaces.isEmpty()) {
            throw IllegalStateException(
                "No face found in source image"
            )
        }

        if (targetFaces.isEmpty()) {
            throw IllegalStateException(
                "No face found in target image"
            )
        }

        /*
         * Use the largest face in each image.
         */
        val sourceFace =
            sourceFaces.maxByOrNull {
                it.area()
            }
                ?: throw IllegalStateException(
                    "Could not select source face"
                )

        val targetFace =
            targetFaces.maxByOrNull {
                it.area()
            }
                ?: throw IllegalStateException(
                    "Could not select target face"
                )

        /*
         * Get 478 landmarks.
         */
        val sourceLandmarks =
            detectLandmarks(
                source,
                sourceFace
            )

        val targetLandmarks =
            detectLandmarks(
                target,
                targetFace
            )

        if (sourceLandmarks.size < 400) {
            throw IllegalStateException(
                "Source face landmarks unavailable"
            )
        }

        if (targetLandmarks.size < 400) {
            throw IllegalStateException(
                "Target face landmarks unavailable"
            )
        }

        /*
         * Align both faces to the HyperSwap 256x256 template.
         */
        val sourcePrepared =
            prepareFace(
                source,
                sourceFace,
                sourceLandmarks
            )

        val targetPrepared =
            prepareFace(
                target,
                targetFace,
                targetLandmarks
            )

        /*
         * ArcFace source identity embedding.
         */
        val sourceEmbedding =
            createArcFaceEmbedding(
                sourcePrepared.bitmap
            )

        /*
         * HyperSwap.
         */
        val swapResult =
            runHyperSwap(
                targetPrepared.bitmap,
                sourceEmbedding
            )

        /*
         * IMPORTANT:
         *
         * Create a target-specific blending mask from all
         * 478 target landmarks.
         *
         * This is what makes the new version adaptive.
         */
        val targetFaceMask =
            createTargetFaceMask(
                targetPrepared.alignedLandmarks
            )

        /*
         * Paste the HyperSwap result back using:
         *
         * HyperSwap mask
         *        +
         * adaptive expansion
         *        +
         * target landmark boundary
         *        +
         * feathering
         */
        return pasteBack(
            originalTarget = target,
            targetPrepared = targetPrepared,
            swapResult = swapResult,
            targetFaceMask = targetFaceMask
        )
    }

    private fun detectFaces(
        bitmap: Bitmap
    ): List<DetectedFace> {

        val session =
            detectorSession
                ?: throw IllegalStateException(
                    "Face detector not loaded"
                )

        /*
         * The existing detector implementation in the project
         * is expected to provide this preprocessing/output.
         *
         * We keep the detector interface unchanged.
         */
        return FaceDetection.detect(
            bitmap,
            session,
            environment
        )
    }

    private fun detectLandmarks(
        bitmap: Bitmap,
        face: DetectedFace
    ): Array<FloatArray> {

        val session =
            landmarkerSession
                ?: throw IllegalStateException(
                    "Face landmarker not loaded"
                )

        return FaceLandmarker.detect(
            bitmap,
            face,
            session,
            environment
        )
    }

    private fun prepareFace(
        bitmap: Bitmap,
        face: DetectedFace,
        landmarks: Array<FloatArray>
    ): PreparedFace {

        /*
         * Use the same 1.5x face ROI strategy as the working version.
         */
        val faceRect =
            face.rect

        val centerX =
            faceRect.centerX()

        val centerY =
            faceRect.centerY()

        val width =
            faceRect.width()

        val height =
            faceRect.height()

        val cropWidth =
            width * 1.5f

        val cropHeight =
            height * 1.5f

        val left =
            (centerX - cropWidth / 2f)
                .coerceAtLeast(0f)

        val top =
            (centerY - cropHeight / 2f)
                .coerceAtLeast(0f)

        val right =
            (centerX + cropWidth / 2f)
                .coerceAtMost(
                    bitmap.width.toFloat()
                )

        val bottom =
            (centerY + cropHeight / 2f)
                .coerceAtMost(
                    bitmap.height.toFloat()
                )

        val cropRect =
            RectF(
                left,
                top,
                right,
                bottom
            )

        val crop =
            Bitmap.createBitmap(
                bitmap,
                cropRect.left.toInt(),
                cropRect.top.toInt(),
                cropRect.width().toInt()
                    .coerceAtLeast(1),
                cropRect.height().toInt()
                    .coerceAtLeast(1)
            )

        /*
         * Convert original-image landmarks into crop coordinates.
         */
        val cropLandmarks =
            Array(
                landmarks.size
            ) { index ->

                floatArrayOf(
                    landmarks[index][0] -
                            cropRect.left,

                    landmarks[index][1] -
                            cropRect.top
                )
            }

        /*
         * Five-point alignment.
         */
        val fivePoints =
            FaceAlignment.fivePoints(
                cropLandmarks
            )

        val template =
            arrayOf(
                floatArrayOf(
                    84.87f,
                    105.94f
                ),
                floatArrayOf(
                    171.13f,
                    105.94f
                ),
                floatArrayOf(
                    128.0f,
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

        val transform =
            calculateSimilarityTransform(
                fivePoints,
                template
            )

        val aligned =
            warpBitmap(
                crop,
                transform,
                FACE_SIZE,
                FACE_SIZE
            )

        /*
         * Transform ALL 478 landmarks into aligned 256x256 space.
         *
         * This is the important new part.
         */
        val alignedLandmarks =
            Array(
                cropLandmarks.size
            ) { index ->

                floatArrayOf(
                    cropLandmarks[index][0],
                    cropLandmarks[index][1]
                )
            }

        val landmarkArray =
            FloatArray(
                alignedLandmarks.size * 2
            )

        for (i in alignedLandmarks.indices) {

            landmarkArray[i * 2] =
                alignedLandmarks[i][0]

            landmarkArray[i * 2 + 1] =
                alignedLandmarks[i][1]
        }

        transform.mapPoints(
            landmarkArray
        )

        for (i in alignedLandmarks.indices) {

            alignedLandmarks[i][0] =
                landmarkArray[i * 2]

            alignedLandmarks[i][1] =
                landmarkArray[i * 2 + 1]
        }

        crop.recycle()

        return PreparedFace(
            bitmap = aligned,
            transform = transform,
            alignedLandmarks = alignedLandmarks
        )
    }

    private fun warpBitmap(
        source: Bitmap,
        transform: Matrix,
        width: Int,
        height: Int
    ): Bitmap {

        val output =
            Bitmap.createBitmap(
                width,
                height,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(output)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            source,
            transform,
            paint
        )

        return output
    }

    private fun calculateSimilarityTransform(
        sourcePoints: Array<FloatArray>,
        targetPoints: Array<FloatArray>
    ): Matrix {

        if (sourcePoints.size != 5 ||
            targetPoints.size != 5
        ) {
            throw IllegalArgumentException(
                "Exactly five points are required"
            )
        }

        /*
         * Least-squares similarity transform.
         *
         * x' = a*x - b*y + tx
         * y' = b*x + a*y + ty
         */
        var srcMeanX =
            0.0

        var srcMeanY =
            0.0

        var dstMeanX =
            0.0

        var dstMeanY =
            0.0

        for (i in 0 until 5) {

            srcMeanX +=
                sourcePoints[i][0]

            srcMeanY +=
                sourcePoints[i][1]

            dstMeanX +=
                targetPoints[i][0]

            dstMeanY +=
                targetPoints[i][1]
        }

        srcMeanX /= 5.0
        srcMeanY /= 5.0

        dstMeanX /= 5.0
        dstMeanY /= 5.0

        var numA =
            0.0

        var numB =
            0.0

        var denominator =
            0.0

        for (i in 0 until 5) {

            val sx =
                sourcePoints[i][0] -
                        srcMeanX

            val sy =
                sourcePoints[i][1] -
                        srcMeanY

            val dx =
                targetPoints[i][0] -
                        dstMeanX

            val dy =
                targetPoints[i][1] -
                        dstMeanY

            numA +=
                sx * dx +
                        sy * dy

            numB +=
                sx * dy -
                        sy * dx

            denominator +=
                sx * sx +
                        sy * sy
        }

        if (denominator < 1e-8) {
            throw IllegalStateException(
                "Invalid face alignment"
            )
        }

        val a =
            numA / denominator

        val b =
            numB / denominator

        val tx =
            dstMeanX -
                    a * srcMeanX +
                    b * srcMeanY

        val ty =
            dstMeanY -
                    b * srcMeanX -
                    a * srcMeanY

        return Matrix().apply {

            setValues(
                floatArrayOf(
                    a.toFloat(),
                    (-b).toFloat(),
                    tx.toFloat(),

                    b.toFloat(),
                    a.toFloat(),
                    ty.toFloat(),

                    0f,
                    0f,
                    1f
                )
            )
        }
    }

    private fun createArcFaceEmbedding(
        bitmap: Bitmap
    ): FloatArray {

        val session =
            arcFaceSession
                ?: throw IllegalStateException(
                    "ArcFace model not loaded"
                )

        val input =
            Bitmap.createScaledBitmap(
                bitmap,
                ARCFACE_SIZE,
                ARCFACE_SIZE,
                true
            )

        val data =
            FloatArray(
                1 * 3 *
                        ARCFACE_SIZE *
                        ARCFACE_SIZE
            )

        var index =
            0

        /*
         * RGB CHW, normalized [-1, 1].
         */
        for (channel in 0 until 3) {

            for (y in 0 until ARCFACE_SIZE) {

                for (x in 0 until ARCFACE_SIZE) {

                    val pixel =
                        input.getPixel(
                            x,
                            y
                        )

                    val value =
                        when (channel) {

                            0 ->
                                Color.red(
                                    pixel
                                )

                            1 ->
                                Color.green(
                                    pixel
                                )

                            else ->
                                Color.blue(
                                    pixel
                                )
                        }

                    data[index++] =
                        value / 127.5f -
                                1.0f
                }
            }
        }

        input.recycle()

        val inputName =
            session.inputNames
                .first()

        val tensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(data),
                longArrayOf(
                    1,
                    3,
                    ARCFACE_SIZE.toLong(),
                    ARCFACE_SIZE.toLong()
                )
            )

        tensor.use {

            session.run(
                mapOf(
                    inputName to tensor
                )
            ).use { results ->

                val output =
                    extractFloatArray(
                        results[0].value
                    )

                if (output.isEmpty()) {
                    throw IllegalStateException(
                        "ArcFace returned empty embedding"
                    )
                }

                /*
                 * L2 normalize.
                 */
                var norm =
                    0.0

                for (v in output) {
                    norm +=
                        v.toDouble() *
                                v.toDouble()
                }

                norm =
                    sqrt(norm)

                if (norm < 1e-12) {
                    throw IllegalStateException(
                        "Invalid ArcFace embedding"
                    )
                }

                return FloatArray(
                    output.size
                ) { i ->
                    (
                        output[i] /
                                norm
                        ).toFloat()
                }
            }
        }
    }

    private fun runHyperSwap(
        target: Bitmap,
        sourceEmbedding: FloatArray
    ): HyperSwapResult {

        val session =
            hyperSwapSession
                ?: throw IllegalStateException(
                    "HyperSwap model not loaded"
                )

        val imageData =
            FloatArray(
                1 * 3 *
                        FACE_SIZE *
                        FACE_SIZE
            )

        var index =
            0

        /*
         * RGB CHW, normalized [-1, 1].
         */
        for (channel in 0 until 3) {

            for (y in 0 until FACE_SIZE) {

                for (x in 0 until FACE_SIZE) {

                    val pixel =
                        target.getPixel(
                            x,
                            y
                        )

                    val value =
                        when (channel) {

                            0 ->
                                Color.red(
                                    pixel
                                )

                            1 ->
                                Color.green(
                                    pixel
                                )

                            else ->
                                Color.blue(
                                    pixel
                                )
                        }

                    imageData[index++] =
                        value / 127.5f -
                                1.0f
                }
            }
        }

        val imageTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(
                    imageData
                ),
                longArrayOf(
                    1,
                    3,
                    FACE_SIZE.toLong(),
                    FACE_SIZE.toLong()
                )
            )

        val embeddingTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(
                    sourceEmbedding
                ),
                longArrayOf(
                    1,
                    sourceEmbedding.size.toLong()
                )
            )

        imageTensor.use {
            embeddingTensor.use {

                val inputNames =
                    session.inputNames
                        .toList()

                if (inputNames.size < 2) {
                    throw IllegalStateException(
                        "Unexpected HyperSwap inputs"
                    )
                }

                /*
                 * HyperSwap 1a expects:
                 *
                 * image:  [1,3,256,256]
                 * source: [1,512]
                 */
                val inputs =
                    HashMap<String, OnnxTensor>()

                inputs[
                    inputNames[0]
                ] =
                    imageTensor

                inputs[
                    inputNames[1]
                ] =
                    embeddingTensor

                session.run(
                    inputs
                ).use { results ->

                    if (results.size < 2) {
                        throw IllegalStateException(
                            "HyperSwap returned insufficient outputs"
                        )
                    }

                    val imageOutput =
                        extractFloatArray(
                            results[0].value
                        )

                    val maskOutput =
                        extractFloatArray(
                            results[1].value
                        )

                    val outputBitmap =
                        imageToBitmap(
                            imageOutput
                        )

                    return HyperSwapResult(
                        bitmap = outputBitmap,
                        mask = maskOutput
                    )
                }
            }
        }
    }

    /*
     * ============================================================
     * ADAPTIVE TARGET FACE MASK
     * ============================================================
     */

    private fun createTargetFaceMask(
        landmarks: Array<FloatArray>
    ): FloatArray {

        val points =
            ArrayList<Point2D>(
                landmarks.size
            )

        for (landmark in landmarks) {

            if (landmark.size < 2) {
                continue
            }

            val x =
                landmark[0]

            val y =
                landmark[1]

            if (
                x.isFinite() &&
                y.isFinite()
            ) {

                if (
                    x >= -10f &&
                    x <= FACE_SIZE + 10f &&
                    y >= -10f &&
                    y <= FACE_SIZE + 10f
                ) {

                    points.add(
                        Point2D(
                            x.coerceIn(
                                0f,
                                FACE_SIZE - 1f
                            ),
                            y.coerceIn(
                                0f,
                                FACE_SIZE - 1f
                            )
                        )
                    )
                }
            }
        }

        if (points.size < 10) {

            /*
             * Fallback to a broad face-shaped ellipse.
             */
            return createFallbackFaceMask()
        }

        val hull =
            convexHull(
                points
            )

        if (hull.size < 3) {
            return createFallbackFaceMask()
        }

        /*
         * Find hull centroid.
         */
        var centerX =
            0f

        var centerY =
            0f

        for (point in hull) {

            centerX +=
                point.x

            centerY +=
                point.y
        }

        centerX /=
            hull.size

        centerY /=
            hull.size

        /*
         * Slightly shrink the hull.
         *
         * This keeps the target-specific shape but removes
         * a tiny amount around the extreme boundary.
         */
        val shrunkHull =
            hull.map { point ->

                val dx =
                    point.x -
                            centerX

                val dy =
                    point.y -
                            centerY

                val distance =
                    sqrt(
                        dx * dx +
                                dy * dy
                    )

                if (distance < 1f) {

                    Point2D(
                        point.x,
                        point.y
                    )

                } else {

                    val scale =
                        max(
                            0f,
                            (
                                distance -
                                        HULL_SHRINK
                                ) /
                                    distance
                        )

                    Point2D(
                        centerX +
                                dx * scale,

                        centerY +
                                dy * scale
                    )
                }
            }

        /*
         * Rasterize the polygon.
         */
        val bitmap =
            Bitmap.createBitmap(
                FACE_SIZE,
                FACE_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(bitmap)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG
            )

        paint.color =
            Color.WHITE

        paint.style =
            Paint.Style.FILL

        val path =
            Path()

        path.moveTo(
            shrunkHull[0].x,
            shrunkHull[0].y
        )

        for (i in 1 until shrunkHull.size) {

            path.lineTo(
                shrunkHull[i].x,
                shrunkHull[i].y
            )
        }

        path.close()

        canvas.drawPath(
            path,
            paint
        )

        val pixels =
            IntArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        bitmap.getPixels(
            pixels,
            0,
            FACE_SIZE,
            0,
            0,
            FACE_SIZE,
            FACE_SIZE
        )

        val mask =
            FloatArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        for (i in pixels.indices) {

            mask[i] =
                Color.alpha(
                    pixels[i]
                ) / 255f
        }

        bitmap.recycle()

        return mask
    }

    private fun createFallbackFaceMask():
            FloatArray {

        val mask =
            FloatArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        val cx =
            FACE_SIZE / 2f

        val cy =
            139f

        val rx =
            91f

        val ry =
            108f

        for (y in 0 until FACE_SIZE) {

            for (x in 0 until FACE_SIZE) {

                val dx =
                    (x - cx) /
                            rx

                val dy =
                    (y - cy) /
                            ry

                val distance =
                    dx * dx +
                            dy * dy

                if (distance <= 1f) {

                    mask[
                        y * FACE_SIZE + x
                    ] = 1f
                }
            }
        }

        return mask
    }

    private fun convexHull(
        input: List<Point2D>
    ): List<Point2D> {

        if (input.size <= 2) {
            return input
        }

        val points =
            input
                .distinctBy {
                    "${it.x}_${it.y}"
                }
                .sortedWith(
                    compareBy<Point2D> {
                        it.x
                    }.thenBy {
                        it.y
                    }
                )

        if (points.size <= 2) {
            return points
        }

        val lower =
            ArrayList<Point2D>()

        for (point in points) {

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
            ArrayList<Point2D>()

        for (i in points.indices.reversed()) {

            val point =
                points[i]

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

    private fun cross(
        a: Point2D,
        b: Point2D,
        c: Point2D
    ): Float {

        return (
            b.x - a.x
            ) * (
            c.y - a.y
            ) -
                (
                    b.y - a.y
                    ) * (
                    c.x - a.x
                    )
    }

    /*
     * ============================================================
     * ADAPTIVE BLENDING
     * ============================================================
     */

    private fun pasteBack(
        originalTarget: Bitmap,
        targetPrepared: PreparedFace,
        swapResult: HyperSwapResult,
        targetFaceMask: FloatArray
    ): Bitmap {

        /*
         * Determine how much the HyperSwap mask needs to expand
         * relative to the actual target face.
         */
        val expansion =
            calculateAdaptiveExpansion(
                swapResult.mask,
                targetFaceMask
            )

        val expandedMask =
            expandMask(
                swapResult.mask,
                expansion.first,
                expansion.second
            )

        /*
         * Clip the expanded generated-face mask to the actual
         * target face shape.
         *
         * This is the key protection against:
         *
         * thick source -> thin target
         */
        val clippedMask =
            FloatArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        for (i in clippedMask.indices) {

            clippedMask[i] =
                min(
                    expandedMask[i],
                    targetFaceMask[i]
                )
        }

        /*
         * Feather the boundary.
         */
        val blurredMask =
            blurMask(
                clippedMask,
                MASK_BLUR_RADIUS
            )

        /*
         * Convert to a softly weighted alpha mask.
         */
        val finalMask =
            FloatArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        for (i in finalMask.indices) {

            val value =
                blurredMask[i]
                    .coerceIn(
                        0f,
                        1f
                    )

            /*
             * Smoothstep.
             */
            val smooth =
                value * value *
                        (
                            3f -
                                    2f * value
                            )

            finalMask[i] =
                (
                    smooth *
                            MAX_ALPHA
                    ).coerceIn(
                        0f,
                        MAX_ALPHA
                    )
        }

        val maskedSwap =
            createMaskedSwap(
                swapResult.bitmap,
                finalMask
            )

        /*
         * Invert the alignment transform.
         */
        val inverse =
            Matrix()

        if (
            !targetPrepared.transform.invert(
                inverse
            )
        ) {

            maskedSwap.recycle()

            throw IllegalStateException(
                "Could not invert face transform"
            )
        }

        val output =
            originalTarget.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val canvas =
            Canvas(output)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            maskedSwap,
            inverse,
            paint
        )

        maskedSwap.recycle()

        if (
            targetPrepared.bitmap !== output
        ) {
            targetPrepared.bitmap.recycle()
        }

        swapResult.bitmap.recycle()

        return output
    }

    private fun calculateAdaptiveExpansion(
        swapMask: FloatArray,
        targetMask: FloatArray
    ): Pair<Int, Int> {

        val swapBounds =
            findMaskBounds(
                swapMask,
                0.15f
            )

        val targetBounds =
            findMaskBounds(
                targetMask,
                0.50f
            )

        if (
            swapBounds == null ||
            targetBounds == null
        ) {

            return Pair(
                MIN_EXPAND,
                MIN_EXPAND
            )
        }

        val swapWidth =
            swapBounds.width()

        val swapHeight =
            swapBounds.height()

        val targetWidth =
            targetBounds.width()

        val targetHeight =
            targetBounds.height()

        /*
         * If the target face is wider than the HyperSwap mask,
         * expand toward it.
         *
         * If the target is narrower, expansion stays minimal.
         */
        val extraWidth =
            max(
                0f,
                targetWidth -
                        swapWidth
            )

        val extraHeight =
            max(
                0f,
                targetHeight -
                        swapHeight
            )

        val expandX =
            (
                extraWidth * 0.28f
                )
                .toInt()
                .coerceIn(
                    MIN_EXPAND,
                    MAX_EXPAND_X
                )

        val expandY =
            (
                extraHeight * 0.28f
                )
                .toInt()
                .coerceIn(
                    MIN_EXPAND,
                    MAX_EXPAND_Y
                )

        return Pair(
            expandX,
            expandY
        )
    }

    private fun findMaskBounds(
        mask: FloatArray,
        threshold: Float
    ): MaskBounds? {

        var minX =
            FACE_SIZE

        var minY =
            FACE_SIZE

        var maxX =
            -1

        var maxY =
            -1

        for (y in 0 until FACE_SIZE) {

            for (x in 0 until FACE_SIZE) {

                val index =
                    y * FACE_SIZE + x

                if (
                    index < mask.size &&
                    mask[index] >
                        threshold
                ) {

                    minX =
                        min(
                            minX,
                            x
                        )

                    minY =
                        min(
                            minY,
                            y
                        )

                    maxX =
                        max(
                            maxX,
                            x
                        )

                    maxY =
                        max(
                            maxY,
                            y
                        )
                }
            }
        }

        if (maxX < minX ||
            maxY < minY
        ) {
            return null
        }

        return MaskBounds(
            minX,
            minY,
            maxX,
            maxY
        )
    }

    private fun expandMask(
        mask: FloatArray,
        radiusX: Int,
        radiusY: Int
    ): FloatArray {

        if (
            radiusX <= 0 &&
            radiusY <= 0
        ) {
            return mask.copyOf()
        }

        val output =
            FloatArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        /*
         * Elliptical max filter.
         *
         * This expands the mask while retaining its shape better
         * than simply drawing a large rectangle.
         */
        for (y in 0 until FACE_SIZE) {

            for (x in 0 until FACE_SIZE) {

                var maximum =
                    0f

                val top =
                    max(
                        0,
                        y - radiusY
                    )

                val bottom =
                    min(
                        FACE_SIZE - 1,
                        y + radiusY
                    )

                val left =
                    max(
                        0,
                        x - radiusX
                    )

                val right =
                    min(
                        FACE_SIZE - 1,
                        x + radiusX
                    )

                for (ny in top..bottom) {

                    val dy =
                        ny - y

                    for (nx in left..right) {

                        val dx =
                            nx - x

                        val normalizedX =
                            if (
                                radiusX == 0
                            ) {
                                0f
                            } else {
                                dx.toFloat() /
                                        radiusX
                            }

                        val normalizedY =
                            if (
                                radiusY == 0
                            ) {
                                0f
                            } else {
                                dy.toFloat() /
                                        radiusY
                            }

                        if (
                            normalizedX *
                                normalizedX +
                            normalizedY *
                                normalizedY <=
                                1f
                        ) {

                            val value =
                                mask[
                                    ny *
                                            FACE_SIZE +
                                            nx
                                ]

                            if (
                                value >
                                maximum
                            ) {
                                maximum =
                                    value
                            }
                        }
                    }
                }

                output[
                    y * FACE_SIZE + x
                ] =
                    maximum
            }
        }

        return output
    }

    private fun blurMask(
        input: FloatArray,
        radius: Int
    ): FloatArray {

        if (radius <= 0) {
            return input.copyOf()
        }

        val horizontal =
            FloatArray(
                input.size
            )

        val output =
            FloatArray(
                input.size
            )

        /*
         * Horizontal box blur.
         */
        for (y in 0 until FACE_SIZE) {

            for (x in 0 until FACE_SIZE) {

                var sum =
                    0f

                var count =
                    0

                val start =
                    max(
                        0,
                        x - radius
                    )

                val end =
                    min(
                        FACE_SIZE - 1,
                        x + radius
                    )

                for (xx in start..end) {

                    sum +=
                        input[
                            y *
                                    FACE_SIZE +
                                    xx
                        ]

                    count++
                }

                horizontal[
                    y *
                            FACE_SIZE +
                            x
                ] =
                    sum /
                            count
            }
        }

        /*
         * Vertical box blur.
         */
        for (y in 0 until FACE_SIZE) {

            for (x in 0 until FACE_SIZE) {

                var sum =
                    0f

                var count =
                    0

                val start =
                    max(
                        0,
                        y - radius
                    )

                val end =
                    min(
                        FACE_SIZE - 1,
                        y + radius
                    )

                for (yy in start..end) {

                    sum +=
                        horizontal[
                            yy *
                                    FACE_SIZE +
                                    x
                        ]

                    count++
                }

                output[
                    y *
                            FACE_SIZE +
                            x
                ] =
                    sum /
                            count
            }
        }

        return output
    }

    private fun createMaskedSwap(
        bitmap: Bitmap,
        mask: FloatArray
    ): Bitmap {

        val output =
            Bitmap.createBitmap(
                FACE_SIZE,
                FACE_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val pixels =
            IntArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        bitmap.getPixels(
            pixels,
            0,
            FACE_SIZE,
            0,
            0,
            FACE_SIZE,
            FACE_SIZE
        )

        for (i in pixels.indices) {

            val alpha =
                (
                    mask.getOrElse(
                        i
                    ) {
                        0f
                    } * 255f
                    )
                    .toInt()
                    .coerceIn(
                        0,
                        255
                    )

            pixels[i] =
                Color.argb(
                    alpha,
                    Color.red(
                        pixels[i]
                    ),
                    Color.green(
                        pixels[i]
                    ),
                    Color.blue(
                        pixels[i]
                    )
                )
        }

        output.setPixels(
            pixels,
            0,
            FACE_SIZE,
            0,
            0,
            FACE_SIZE,
            FACE_SIZE
        )

        return output
    }

    /*
     * ============================================================
     * OUTPUT CONVERSION
     * ============================================================
     */

    private fun imageToBitmap(
        values: FloatArray
    ): Bitmap {

        if (
            values.size <
            FACE_SIZE *
            FACE_SIZE *
            3
        ) {
            throw IllegalStateException(
                "Invalid HyperSwap image output"
            )
        }

        val bitmap =
            Bitmap.createBitmap(
                FACE_SIZE,
                FACE_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val pixels =
            IntArray(
                FACE_SIZE *
                        FACE_SIZE
            )

        /*
         * HyperSwap output is:
         *
         * [1,3,256,256]
         *
         * RGB CHW, [-1,1].
         */
        val plane =
            FACE_SIZE *
                    FACE_SIZE

        for (y in 0 until FACE_SIZE) {

            for (x in 0 until FACE_SIZE) {

                val pixelIndex =
                    y *
                            FACE_SIZE +
                            x

                val r =
                    values[
                        pixelIndex
                    ]

                val g =
                    values[
                        plane +
                                pixelIndex
                    ]

                val b =
                    values[
                        plane * 2 +
                                pixelIndex
                    ]

                val red =
                    (
                        (
                            r + 1f
                            ) *
                                127.5f
                        )
                        .toInt()
                        .coerceIn(
                            0,
                            255
                        )

                val green =
                    (
                        (
                            g + 1f
                            ) *
                                127.5f
                        )
                        .toInt()
                        .coerceIn(
                            0,
                            255
                        )

                val blue =
                    (
                        (
                            b + 1f
                            ) *
                                127.5f
                        )
                        .toInt()
                        .coerceIn(
                            0,
                            255
                        )

                pixels[pixelIndex] =
                    Color.rgb(
                        red,
                        green,
                        blue
                    )
            }
        }

        bitmap.setPixels(
            pixels,
            0,
            FACE_SIZE,
            0,
            0,
            FACE_SIZE,
            FACE_SIZE
        )

        return bitmap
    }

    private fun extractFloatArray(
        value: Any?
    ): FloatArray {

        when (value) {

            is FloatArray ->
                return value

            is Array<*> -> {

                val result =
                    ArrayList<Float>()

                fun flatten(
                    item: Any?
                ) {

                    when (item) {

                        is Float ->
                            result.add(
                                item
                            )

                        is Double ->
                            result.add(
                                item.toFloat()
                            )

                        is Number ->
                            result.add(
                                item.toFloat()
                            )

                        is FloatArray -> {

                            for (v in item) {
                                result.add(v)
                            }
                        }

                        is Array<*> -> {

                            for (v in item) {
                                flatten(v)
                            }
                        }

                        else -> {
                            // Ignore unsupported values.
                        }
                    }
                }

                flatten(value)

                return result.toFloatArray()
            }

            is Number ->
                return floatArrayOf(
                    value.toFloat()
                )

            else ->
                throw IllegalStateException(
                    "Unsupported ONNX output type: ${
                        value?.javaClass?.name
                    }"
                )
        }
    }

    fun close() {

        if (closed) {
            return
        }

        closed =
            true

        try {
            detectorSession?.close()
        } catch (_: Throwable) {
        }

        try {
            landmarkerSession?.close()
        } catch (_: Throwable) {
        }

        try {
            arcFaceSession?.close()
        } catch (_: Throwable) {
        }

        try {
            hyperSwapSession?.close()
        } catch (_: Throwable) {
        }

        detectorSession =
            null

        landmarkerSession =
            null

        arcFaceSession =
            null

        hyperSwapSession =
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
        val alignedLandmarks: Array<FloatArray>
    )

    private data class HyperSwapResult(
        val bitmap: Bitmap,
        val mask: FloatArray
    )

    private data class Point2D(
        val x: Float,
        val y: Float
    )

    private data class MaskBounds(
        val minX: Int,
        val minY: Int,
        val maxX: Int,
        val maxY: Int
    ) {

        fun width(): Int =
            maxX - minX + 1

        fun height(): Int =
            maxY - minY + 1
    }
}
