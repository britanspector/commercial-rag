import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function CachePage() {
  const item = getNavItemByPath('/cache')!
  return <NavPage item={item} />
}
