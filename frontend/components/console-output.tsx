"use client"

import { Card } from "@/components/ui/card"
import { Terminal } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ConsoleOutputProps {
  logs: string[]
}

export function ConsoleOutput({ logs }: ConsoleOutputProps) {
  const visibleLogs = logs.slice(-50)

  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <Terminal className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold text-foreground">Console</h3>
      </div>

      <div className="bg-muted/30 rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs space-y-1">
        <AnimatePresence initial={false}>
          {visibleLogs.map((log, index) => (
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
