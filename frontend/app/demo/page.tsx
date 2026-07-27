import Link from "next/link";
import { DemoNav } from "@/components/DemoNav";
import { DEMO_MODELS } from "@/lib/demoModels";

/**
 * Demo Hub: lists every model demo from DEMO_MODELS as a card. Adding
 * a model to that config is the only change needed for it to appear
 * here too -- this page has no per-model logic of its own.
 */
export default function DemoHubPage() {
  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-zinc-950">
      <DemoNav />
      <main className="flex w-full max-w-4xl flex-1 flex-col items-center gap-14 px-6 py-20 sm:px-10 sm:py-24">
        <div className="flex flex-col items-center gap-4 text-center">
          <span className="rounded-full border border-black/[.06] bg-black/[.02] px-3 py-1 text-xs font-medium tracking-wide text-zinc-500 dark:border-white/[.08] dark:bg-white/[.04] dark:text-zinc-400">
            AI Demo Suite
          </span>
          <h1 className="text-4xl font-semibold tracking-tight text-black sm:text-5xl dark:text-zinc-50">
            Khmer Sign Language AI Demo
          </h1>
          <p className="max-w-md text-lg leading-relaxed text-zinc-600 dark:text-zinc-400">
            Choose a model to try.
          </p>
        </div>
        <div className="grid w-full grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {DEMO_MODELS.map((model) => (
            <div
              key={model.id}
              className="group flex flex-col gap-4 rounded-2xl border border-black/[.06] bg-white p-6 text-left shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md dark:border-white/[.08] dark:bg-zinc-900/60 dark:hover:border-white/[.14]"
            >
              <h2 className="text-lg font-semibold text-black dark:text-zinc-50">
                {model.fullName}
              </h2>
              <p className="flex-1 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
                {model.description}
              </p>
              {model.href ? (
                <Link
                  href={model.href}
                  className="self-start rounded-full bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
                >
                  Open Demo
                </Link>
              ) : (
                <span className="self-start rounded-full border border-dashed border-black/10 px-4 py-2 text-sm font-medium text-zinc-400 dark:border-white/10 dark:text-zinc-600">
                  Coming soon
                </span>
              )}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
