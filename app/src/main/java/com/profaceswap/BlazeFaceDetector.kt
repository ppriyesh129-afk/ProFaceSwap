package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class BlazeFaceDetector(
    private val session: OrtSession
) {

    private val environment =
        OrtEnvironment.getEnvironment()

    companion object {

        private const val INPUT_SIZE = 128
        private const val ANCHOR_COUNT = 896
        private const val SCORE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.3f
        private const val KEYPOINT_COUNT = 6
    }

    fun detect(bitmap: Bitmap): List<BlazeFaceResult> {

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        if (originalWidth <= 0 || originalHeight <= 0) {
            return emptyList()
        }

        val scale =
            INPUT_SIZE.toFloat() /
                    max(
                        originalWidth,
                        originalHeight
                    ).toFloat()

        val resizedWidth =
            originalWidth * scale

        val resizedHeight =
            originalHeight * scale

        val padX =
            (INPUT_SIZE - resizedWidth) / 2f

        val padY =
            (INPUT_SIZE - resizedHeight) / 2f

        val matrix =
            Matrix().apply {

                postScale(
                    scale,
                    scale
                )

                postTranslate(
                    padX,
                    padY
                )
            }

        val inputBitmap =
            Bitmap.createBitmap(
                INPUT_SIZE,
                INPUT_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(inputBitmap)

        canvas.drawBitmap(
            bitmap,
            matrix,
            null
        )

        val inputBuffer =
            createInput(inputBitmap)

        val tensor =
            OnnxTensor.createTensor(
                environment,
                inputBuffer,
                longArrayOf(
                    1,
                    3,
                    INPUT_SIZE.toLong(),
                    INPUT_SIZE.toLong()
                )
            )

        val result =
            session.run(
                mapOf(
                    session.inputNames.first()
                        to tensor
                )
            )

        val regressors =
            extractFloat2D(
                result[0].value
            )

        val scores =
            extractFloat2D(
                result[1].value
            )

        val anchors =
            generateAnchors()

        val candidates =
            mutableListOf<Detection>()

        for (i in 0 until ANCHOR_COUNT) {

            if (i >= regressors.size ||
                i >= scores.size
            ) {
                break
            }

            if (regressors[i].size < 16 ||
                scores[i].isEmpty()
            ) {
                continue
            }

            val score =
                sigmoid(
                    scores[i][0]
                )

            if (score < SCORE_THRESHOLD) {
                continue
            }

            val values =
                regressors[i]

            val anchor =
                anchors[i]

            val cx =
                values[0] /
                        INPUT_SIZE +
                        anchor[0]

            val cy =
                values[1] /
                        INPUT_SIZE +
                        anchor[1]

            val width =
                values[2] /
                        INPUT_SIZE

            val height =
                values[3] /
                        INPUT_SIZE

            val x1 =
                cx - width / 2f

            val y1 =
                cy - height / 2f

            val x2 =
                cx + width / 2f

            val y2 =
                cy + height / 2f

            val keypoints =
                Array(
                    KEYPOINT_COUNT
                ) { keypointIndex ->

                    val offset =
                        4 + keypointIndex * 2

                    floatArrayOf(
                        values[offset] /
                                INPUT_SIZE +
                                anchor[0],

                        values[offset + 1] /
                                INPUT_SIZE +
                                anchor[1]
                    )
                }

            candidates.add(
                Detection(
                    score = score,
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    keypoints = keypoints
                )
            )
        }

        tensor.close()
        result.close()
        inputBitmap.recycle()

        val selected =
            weightedNms(
                candidates
            )

        return selected.map { detection ->

            val x1 =
                unletterbox(
                    detection.x1,
                    padX,
                    scale,
                    originalWidth.toFloat()
                )

            val y1 =
                unletterbox(
                    detection.y1,
                    padY,
                    scale,
                    originalHeight.toFloat()
                )

            val x2 =
                unletterbox(
                    detection.x2,
                    padX,
                    scale,
                    originalWidth.toFloat()
                )

            val y2 =
                unletterbox(
                    detection.y2,
                    padY,
                    scale,
                    originalHeight.toFloat()
                )

            BlazeFaceResult(
                left = x1,
                top = y1,
                right = x2,
                bottom = y2,
                score = detection.score
            )
        }
    }

    private fun generateAnchors(): List<FloatArray> {

        val anchors =
            ArrayList<FloatArray>(
                ANCHOR_COUNT
            )

        // First feature map:
        // 16 x 16 cells
        // 2 anchors per cell
        addGridAnchors(
            anchors = anchors,
            cells = 16,
            repeats = 2
        )

        // Second feature map:
        // 8 x 8 cells
        // 6 anchors per cell
        addGridAnchors(
            anchors = anchors,
            cells = 8,
            repeats = 6
        )

        return anchors
    }

    private fun addGridAnchors(
        anchors: MutableList<FloatArray>,
        cells: Int,
        repeats: Int
    ) {

        for (y in 0 until cells) {

            for (x in 0 until cells) {

                val centerX =
                    (x + 0.5f) /
                            cells.toFloat()

                val centerY =
                    (y + 0.5f) /
                            cells.toFloat()

                repeat(repeats) {

                    anchors.add(
                        floatArrayOf(
                            centerX,
                            centerY
                        )
                    )
                }
            }
        }
    }

    private fun createInput(
        bitmap: Bitmap
    ): FloatBuffer {

        val pixels =
            IntArray(
                INPUT_SIZE *
                        INPUT_SIZE
            )

        bitmap.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val buffer =
            FloatBuffer.allocate(
                3 *
                        INPUT_SIZE *
                        INPUT_SIZE
            )

        // RGB, CHW, [-1, 1]
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
                    (
                        value.toFloat() -
                                127.5f
                        ) / 127.5f
                )
            }
        }

        buffer.rewind()

        return buffer
    }

    private fun extractFloat2D(
        raw: Any
    ): Array<FloatArray> {

        return when (raw) {

            is Array<*> -> {

                val first =
                    raw.firstOrNull()

                when (first) {

                    is Array<*> -> {

                        first.mapNotNull { row ->

                            when (row) {

                                is FloatArray ->
                                    row

                                is DoubleArray ->
                                    FloatArray(
                                        row.size
                                    ) {
                                        row[it]
                                            .toFloat()
                                    }

                                else ->
                                    null
                            }

                        }.toTypedArray()
                    }

                    is FloatArray ->
                        raw.mapNotNull {
                            it as? FloatArray
                        }.toTypedArray()

                    else ->
                        emptyArray()
                }
            }

            is FloatArray ->
                arrayOf(raw)

            else ->
                emptyArray()
        }
    }

    private fun sigmoid(
        value: Float
    ): Float {

        return (
            1.0 /
                    (
                        1.0 +
                                exp(
                                    -value.toDouble()
                                )
                    )
            ).toFloat()
    }

    private fun unletterbox(
        value: Float,
        padding: Float,
        scale: Float,
        maximum: Float
    ): Float {

        return (
            (
                value *
                        INPUT_SIZE -
                        padding
                ) / scale
            ).coerceIn(
                0f,
                maximum
            )
    }

    private fun weightedNms(
        detections: List<Detection>
    ): List<Detection> {

        val remaining =
            detections
                .sortedByDescending {
                    it.score
                }
                .toMutableList()

        val output =
            mutableListOf<Detection>()

        while (remaining.isNotEmpty()) {

            val top =
                remaining.removeAt(0)

            val overlapping =
                mutableListOf<Detection>()

            overlapping.add(top)

            val iterator =
                remaining.iterator()

            while (iterator.hasNext()) {

                val candidate =
                    iterator.next()

                if (
                    iou(
                        top,
                        candidate
                    ) > IOU_THRESHOLD
                ) {

                    overlapping.add(
                        candidate
                    )

                    iterator.remove()
                }
            }

            var totalWeight = 0f

            var x1 = 0f
            var y1 = 0f
            var x2 = 0f
            var y2 = 0f

            var score = 0f

            val keypointSums =
                Array(
                    KEYPOINT_COUNT
                ) {
                    floatArrayOf(
                        0f,
                        0f
                    )
                }

            for (detection in overlapping) {

                val weight =
                    detection.score

                totalWeight += weight

                x1 +=
                    detection.x1 *
                            weight

                y1 +=
                    detection.y1 *
                            weight

                x2 +=
                    detection.x2 *
                            weight

                y2 +=
                    detection.y2 *
                            weight

                score =
                    max(
                        score,
                        detection.score
                    )

                for (k in 0 until KEYPOINT_COUNT) {

                    keypointSums[k][0] +=
                        detection.keypoints[k][0] *
                                weight

                    keypointSums[k][1] +=
                        detection.keypoints[k][1] *
                                weight
                }
            }

            val averagedKeypoints =
                Array(
                    KEYPOINT_COUNT
                ) { k ->

                    floatArrayOf(
                        keypointSums[k][0] /
                                totalWeight,

                        keypointSums[k][1] /
                                totalWeight
                    )
                }

            output.add(
                Detection(
                    score = score,
                    x1 = x1 / totalWeight,
                    y1 = y1 / totalWeight,
                    x2 = x2 / totalWeight,
                    y2 = y2 / totalWeight,
                    keypoints =
                        averagedKeypoints
                )
            )
        }

        return output
    }

    private fun iou(
        a: Detection,
        b: Detection
    ): Float {

        val left =
            max(
                a.x1,
                b.x1
            )

        val top =
            max(
                a.y1,
                b.y1
            )

        val right =
            min(
                a.x2,
                b.x2
            )

        val bottom =
            min(
                a.y2,
                b.y2
            )

        val width =
            max(
                0f,
                right - left
            )

        val height =
            max(
                0f,
                bottom - top
            )

        val intersection =
            width * height

        val areaA =
            max(
                0f,
                a.x2 - a.x1
            ) *
                    max(
                        0f,
                        a.y2 - a.y1
                    )

        val areaB =
            max(
                0f,
                b.x2 - b.x1
            ) *
                    max(
                        0f,
                        b.y2 - b.y1
                    )

        val union =
            areaA +
                    areaB -
                    intersection

        return if (union <= 0f) {
            0f
        } else {
            intersection / union
        }
    }

    private data class Detection(
        val score: Float,
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val keypoints: Array<FloatArray>
    )
}
