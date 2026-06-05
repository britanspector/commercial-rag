import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function WorkflowPage() {
  const item = getNavItemByPath('/workflow')!
  return <NavPage item={item} />
}
