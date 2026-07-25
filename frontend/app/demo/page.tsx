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
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black">
      <DemoNav />
      <main className="flex w-full max-w-3xl flex-1 flex-col items-center gap-10 py-20 px-16">
        <div className="flex flex-col items-center gap-2 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-black dark:text-zinc-50">
            Khmer Sign Language AI Demo
          </h1>
          <p className="max-w-md text-lg text-zinc-600 dark:text-zinc-400">
            Choose a model to try.
          </p>
        </div>
        <div className="grid w-full grid-cols-1 gap-4 sm:grid-cols-2">
          {DEMO_MODELS.map((model) => (
            <div
              key={model.id}
              className="flex flex-col gap-3 rounded-xl border border-black/[.08] bg-white p-6 text-left dark:border-white/[.145] dark:bg-zinc-900"
            >
              <h2 className="text-lg font-semibold text-black dark:text-zinc-50">
                {model.fullName}
              </h2>
              <p className="flex-1 text-sm text-zinc-600 dark:text-zinc-400">
                {model.description}
              </p>
              {model.href ? (
                <Link
                  href={model.href}
                  className="self-start rounded-full border border-black/10 px-4 py-2 text-sm font-medium text-black transition-colors hover:bg-black/5 dark:border-white/20 dark:text-zinc-50 dark:hover:bg-white/10"
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
