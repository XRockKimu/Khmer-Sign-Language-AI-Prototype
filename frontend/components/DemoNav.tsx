"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { DEMO_MODELS } from "@/lib/demoModels";

/**
 * Top-of-page nav bar for every demo page, including the Demo Hub
 * itself. Renders entirely from DEMO_MODELS -- adding a model there is
 * the only change needed for it to show up here too, active/disabled
 * state included. Imported once per page (not duplicated per page's
 * own markup) to keep there being exactly one place this renders from.
 */
export function DemoNav() {
  const pathname = usePathname();

  return (
    <nav className="w-full border-b border-black/[.08] bg-white/80 backdrop-blur-sm dark:border-white/[.145] dark:bg-black/80">
      <div className="mx-auto flex w-full max-w-3xl flex-wrap items-center justify-between gap-3 px-16 py-4">
        <Link
          href="/demo"
          className="text-sm font-semibold text-black dark:text-zinc-50"
        >
          Khmer Sign Language AI Demo
        </Link>
        <div className="flex flex-wrap gap-2">
          {DEMO_MODELS.map((model) => {
            if (!model.href) {
              return (
                <span
                  key={model.id}
                  title="Coming soon"
                  className="cursor-not-allowed rounded-full border border-dashed border-black/10 px-3 py-1 text-xs font-medium text-zinc-400 dark:border-white/10 dark:text-zinc-600"
                >
                  {model.name}
                </span>
              );
            }

            const isActive = pathname === model.href;

            return (
              <Link
                key={model.id}
                href={model.href}
                aria-current={isActive ? "page" : undefined}
                className={
                  isActive
                    ? "rounded-full border border-black bg-black px-3 py-1 text-xs font-medium text-white dark:border-white dark:bg-white dark:text-black"
                    : "rounded-full border border-black/10 px-3 py-1 text-xs font-medium text-black transition-colors hover:bg-black/5 dark:border-white/20 dark:text-zinc-50 dark:hover:bg-white/10"
                }
              >
                {model.name}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
