import { useMemo, useState } from 'react'
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Breadcrumb, Layout, Menu, Typography, theme } from 'antd'
import type { MenuProps } from 'antd'

import { useCurrentNavItem } from '../hooks/useCurrentNavItem'
import {
  APP_TITLE,
} from '../config'
import {
  NAV_GROUP_LABELS,
  NAV_ITEMS,
  type NavGroup,
  type NavItem,
} from '../navigation'

const { Header, Sider, Content } = Layout

function buildMenuItems(): MenuProps['items'] {
  const groups: NavGroup[] = ['overview', 'core', 'future']

  return groups.map((group) => {
    const children = NAV_ITEMS.filter((item) => item.group === group)
    if (group === 'overview') {
      const overview = children[0]
      return overview
        ? {
            key: overview.path,
            icon: overview.icon,
            label: overview.label,
          }
        : null
    }
    return {
      key: group,
      label: NAV_GROUP_LABELS[group],
      type: 'group' as const,
      children: children.map((item) => ({
        key: item.path,
        icon: item.icon,
        label: item.comingSoon ? `${item.label}` : item.label,
        disabled: false,
      })),
    }
  })
}

function buildBreadcrumbItems(current: NavItem | undefined) {
  if (!current || current.path === '/') {
    return [{ title: '概览' }]
  }
  return [
    { title: <Link to="/">概览</Link> },
    { title: NAV_GROUP_LABELS[current.group] },
    { title: current.label },
  ]
}

export function AppLayout() {
  const location = useLocation()
  const navigate = useNavigate()
  const currentNav = useCurrentNavItem()
  const [collapsed, setCollapsed] = useState(false)
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken()

  const menuItems = useMemo(() => buildMenuItems(), [])
  const selectedKeys = useMemo(() => [location.pathname], [location.pathname])
  const breadcrumbItems = useMemo(
    () => buildBreadcrumbItems(currentNav),
    [currentNav],
  )

  const handleMenuClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key)
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider collapsible collapsed={collapsed} onCollapse={setCollapsed} width={240}>
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontWeight: 600,
            padding: '0 12px',
            textAlign: 'center',
          }}
        >
          {collapsed ? 'RAG' : APP_TITLE}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={selectedKeys}
          items={menuItems}
          onClick={handleMenuClick}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            padding: '0 24px',
            background: colorBgContainer,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid rgba(5, 5, 5, 0.06)',
          }}
        >
          <Typography.Title level={4} style={{ margin: 0 }}>
            {currentNav?.label ?? '金融研报 RAG 控制台'}
          </Typography.Title>
          <Breadcrumb items={breadcrumbItems} />
        </Header>
        <Content style={{ margin: 24 }}>
          <div
            style={{
              padding: 24,
              minHeight: 360,
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
            }}
          >
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}
