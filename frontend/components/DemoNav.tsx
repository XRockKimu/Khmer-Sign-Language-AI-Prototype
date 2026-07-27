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
    <nav className="sticky top-0 z-10 w-full border-b border-black/[.06] bg-white/70 backdrop-blur-md dark:border-white/[.08] dark:bg-zinc-950/70">
      <div className="mx-auto flex w-full max-w-4xl flex-wrap items-center justify-between gap-4 px-6 py-4 sm:px-10">
        <Link
          href="/demo"
          className="flex items-center gap-2 text-sm font-semibold tracking-tight text-black transition-opacity hover:opacity-70 dark:text-zinc-50"
        >
          <span className="h-2 w-2 rounded-full bg-indigo-500" />
          Khmer Sign Language AI Demo
        </Link>
        <div className="flex flex-wrap items-center gap-1 rounded-full border border-black/[.06] bg-black/[.02] p-1 dark:border-white/[.08] dark:bg-white/[.04]">
          {DEMO_MODELS.map((model) => {
            if (!model.href) {
              return (
                <span
                  key={model.id}
                  title="Coming soon"
                  className="cursor-not-allowed rounded-full px-3 py-1.5 text-xs font-medium text-zinc-400 dark:text-zinc-600"
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
                    ? "rounded-full bg-black px-3 py-1.5 text-xs font-semibold text-white shadow-sm dark:bg-white dark:text-black"
                    : "rounded-full px-3 py-1.5 text-xs font-medium text-zinc-600 transition-colors hover:bg-black/[.04] hover:text-black dark:text-zinc-400 dark:hover:bg-white/[.08] dark:hover:text-zinc-50"
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
