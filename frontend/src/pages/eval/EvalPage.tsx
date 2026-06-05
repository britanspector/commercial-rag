import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function EvalPage() {
  const item = getNavItemByPath('/eval')!
  return <NavPage item={item} />
}
