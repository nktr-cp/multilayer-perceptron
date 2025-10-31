"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Terminal } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ConsoleOutputProps {
  isTraining: boolean
}

export function ConsoleOutput({ isTraining }: ConsoleOutputProps) {
  const [logs, setLogs] = useState<string[]>(["> Neural network initialized", "> Ready to train"])

  useEffect(() => {
    if (!isTraining) return

    const messages = [
      "Forward pass completed",
      "Computing gradients...",
      "Backpropagation in progress",
      "Weights updated",
      "Epoch completed successfully",
      "Validation accuracy improved",
    ]

    const interval = setInterval(() => {
      const randomMessage = messages[Math.floor(Math.random() * messages.length)]
      const timestamp = new Date().toLocaleTimeString()
      setLogs((prev) => [...prev.slice(-8), `[${timestamp}] ${randomMessage}`])
    }, 2000)

    return () => clearInterval(interval)
  }, [isTraining])

  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <Terminal className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold text-foreground">Console</h3>
      </div>

      <div className="bg-muted/30 rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs space-y-1">
        <AnimatePresence initial={false}>
          {logs.map((log, index) => (
            <motion.div
              key={`${log}-${index}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="text-muted-foreground"
            >
              {log}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </Card>
  )
}
