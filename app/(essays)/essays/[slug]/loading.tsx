export default function BlogPostLoading() {
  return (
    <main className="mx-auto max-w-3xl animate-pulse px-6 pb-24 pt-12 md:px-10 md:pt-16">
      <div className="mb-16 h-4 w-16 rounded bg-neutral-200" />
      <div className="mb-16 space-y-4 text-center">
        <div className="mx-auto h-12 w-3/4 rounded bg-neutral-200" />
        <div className="mx-auto h-5 w-1/2 rounded bg-neutral-100" />
        <div className="mx-auto h-3 w-24 rounded bg-neutral-100" />
      </div>
      <div className="space-y-4">
        <div className="h-4 w-full rounded bg-neutral-100" />
        <div className="h-4 w-full rounded bg-neutral-100" />
        <div className="h-4 w-5/6 rounded bg-neutral-100" />
      </div>
    </main>
  );
}
