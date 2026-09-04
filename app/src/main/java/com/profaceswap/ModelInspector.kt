package com.profaceswap

import ai.onnxruntime.OrtSession

object ModelInspector {

    fun describe(session: OrtSession): String {

        val inputText = buildString {
            append("INPUTS:\n")

            for ((name, info) in session.inputInfo) {
                append(name)
                append(" -> ")
                append(info.info)
                append("\n")
            }
        }

        val outputText = buildString {
            append("OUTPUTS:\n")

            for ((name, info) in session.outputInfo) {
                append(name)
                append(" -> ")
                append(info.info)
                append("\n")
            }
        }

        return inputText + outputText
    }
}
