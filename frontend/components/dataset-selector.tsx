"use client"

import { Card } from "@/components/ui/card"
import { Database } from "lucide-react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface DatasetSelectorProps {
  value: string
  onChange: (value: string) => void
  taskType: "binary" | "multiclass" | "regression"
}

const datasets = {
  binary: [
    { id: "breast-cancer", name: "Breast Cancer", samples: 569, features: 30 },
    { id: "ionosphere", name: "Ionosphere", samples: 351, features: 34 },
  ],
  multiclass: [
    { id: "iris", name: "Iris", samples: 150, features: 4 },
    { id: "wine", name: "Wine", samples: 178, features: 13 },
    { id: "digits", name: "Digits", samples: 1797, features: 64 },
  ],
  regression: [
    { id: "boston", name: "Boston Housing", samples: 506, features: 13 },
    { id: "diabetes", name: "Diabetes", samples: 442, features: 10 },
  ],
}

export function DatasetSelector({ value, onChange, taskType }: DatasetSelectorProps) {
  const availableDatasets = datasets[taskType]

  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <Database className="w-5 h-5 text-secondary" />
        <h3 className="text-lg font-semibold text-foreground">Dataset</h3>
      </div>

      <div className="space-y-2">
        {availableDatasets.map((dataset, index) => (
          <motion.button
            key={dataset.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            onClick={() => onChange(dataset.id)}
            className={cn(
              "w-full p-3 rounded-lg border transition-all text-left",
              value === dataset.id
                ? "border-secondary bg-secondary/10 glow-magenta"
                : "border-border/50 hover:border-secondary/50 hover:bg-muted/30",
            )}
          >
            <p className="font-medium text-foreground">{dataset.name}</p>
            <p className="text-xs text-muted-foreground mt-1">
              {dataset.samples} samples • {dataset.features} features
            </p>
          </motion.button>
        ))}
      </div>
    </Card>
  )
}
