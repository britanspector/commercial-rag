import { PageShell } from '../../components/PageShell'
import { getNavItemByPath } from '../../navigation'
import { CacheStatsPanel } from './CacheStatsPanel'

export function CachePage() {
  const item = getNavItemByPath('/cache')!

  return (
    <PageShell
      title={item.label}
      description={item.description}
      apiEndpoints={item.apiEndpoints}
    >
      <CacheStatsPanel />
    </PageShell>
  )
}
